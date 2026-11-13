// cnn_core.v - sequential int8 CNN engine (one MAC, layer-table driven)
//
// Image (HWC int8) is loaded into buffer A while idle. Layers ping-pong A->B->A...
// Per output value: bias -> sum((x-zin)*w) -> requant (gemmlowp-exact) -> clamp -> store.
// After the last layer it finds the argmax and holds in S_DONE until 'finish'.
// Mirrors verify_twin.py step for step.
`timescale 1ns/1ps
`include "cnn_params.vh"

module cnn_core (
  input  wire             clk,
  input  wire             rst,
  input  wire             start,      // pulse: image is in buffer A
  input  wire             finish,     // pulse: results read out, return to idle
  input  wire             ld_we,      // image load port (only while idle)
  input  wire [`AW-1:0]   ld_addr,
  input  wire [7:0]       ld_data,
  input  wire [`AW-1:0]   ro_addr,    // logit read address (valid in done state)
  output wire [7:0]       ro_data,    // available 2 cycles after ro_addr
  output wire             done,
  output wire [7:0]       class_idx
);

  // ------------------------------------------------------------ states
  localparam [4:0]
    S_IDLE  = 5'd0,  S_LINIT = 5'd1,  S_OINIT = 5'd2,  S_B2   = 5'd3,
    S_B3    = 5'd4,  S_A1    = 5'd5,  S_A2    = 5'd6,  S_A3   = 5'd7,
    S_RD    = 5'd8,  S_MAC   = 5'd9,  S_MUL0  = 5'd10, S_MUL  = 5'd11,
    S_RQ1   = 5'd12, S_RQ2   = 5'd13, S_RQ3   = 5'd14, S_STORE= 5'd15,
    S_LNEXT = 5'd16, S_AG0   = 5'd17, S_AG1   = 5'd18, S_AG2  = 5'd19,
    S_DONE  = 5'd20;

  reg [4:0]          st;
  reg [`LW-1:0]      layer;

  // ------------------------------------------------------------ layer table
  wire [1:0]  t_typ;
  wire [9:0]  t_h, t_w, t_cin, t_oh, t_ow, t_cout, t_k, t_stride, t_padh, t_padw;
  wire [19:0] t_wbase, t_wstride, t_winc;
  wire [15:0] t_bbase;
  wire [7:0]  t_zin, t_zout, t_cmin, t_cmax;

  layer_table lt (
    .idx(layer), .typ(t_typ), .h(t_h), .w(t_w), .cin(t_cin), .oh(t_oh), .ow(t_ow),
    .cout(t_cout), .k(t_k), .stride(t_stride), .padh(t_padh), .padw(t_padw),
    .wbase(t_wbase), .wstride(t_wstride), .winc(t_winc), .bbase(t_bbase),
    .zin(t_zin), .zout(t_zout), .cmin(t_cmin), .cmax(t_cmax)
  );

  // ------------------------------------------------------------ memories
  reg [7:0] memA [0:`ACT_DEPTH-1];
  reg [7:0] memB [0:`ACT_DEPTH-1];
  reg [7:0] rdA, rdB;

  reg signed [7:0]  wrom [0:`WN-1];
  reg signed [31:0] brom [0:`BN-1];
  reg        [31:0] mrom [0:`BN-1];
  reg        [7:0]  erom [0:`BN-1];

  integer ii;
  initial begin
    for (ii = 0; ii < `ACT_DEPTH; ii = ii + 1) begin memA[ii] = 8'd0; memB[ii] = 8'd0; end
    $readmemh("weights_all.mem", wrom);
    $readmemh("bias_all.mem",    brom);
    $readmemh("mult_all.mem",    mrom);
    $readmemh("exp_all.mem",     erom);
  end

  // ------------------------------------------------------------ datapath regs
  reg [9:0]  oy, ox, oc, ky, kx, ic;
  reg [19:0] w_ptr, w_start;
  reg [`AW-1:0] out_ptr;
  reg [`BAW-1:0] bidx;

  reg signed [11:0] iy_r, ix_r;
  reg               valid_r;
  reg signed [19:0] row_r;
  reg [19:0]        in_addr_r;

  reg signed [31:0] acc;
  reg        [31:0] mult_r;
  reg        [7:0]  exp_r;

  reg               acc_neg;
  reg        [63:0] m_sh, m_p;
  reg        [31:0] m_b;
  reg        [5:0]  m_cnt;
  reg signed [63:0] x64;
  reg signed [31:0] q32;
  reg        [7:0]  out_byte;
  reg               wr_en;

  reg               rd_sel, rd_sel_q;
  reg [`AW-1:0]     ag_i;
  reg signed [7:0]  best;
  reg [7:0]         cls;

  reg signed [7:0]  w_rd;
  reg signed [31:0] b_rd;
  reg        [31:0] m_rd;
  reg        [7:0]  e_rd;

  // ------------------------------------------------------------ ROM reads (sync)
  always @(posedge clk) begin
    w_rd <= wrom[w_ptr[`WAW-1:0]];
    b_rd <= brom[bidx];
    m_rd <= mrom[bidx];
    e_rd <= erom[bidx];
  end

  // ------------------------------------------------------------ activation buffers
  wire idle = (st == S_IDLE);
  wire in_ag = (st == S_AG1) || (st == S_AG2);

  wire [`AW-1:0] rd_addr = (st == S_DONE) ? ro_addr :
                           in_ag          ? ag_i    : in_addr_r[`AW-1:0];

  wire weA = idle ? ld_we : (wr_en & layer[0]);
  wire weB = (~idle) & wr_en & (~layer[0]);
  wire [`AW-1:0] waA = idle ? ld_addr : out_ptr;
  wire [7:0]     wdA = idle ? ld_data : out_byte;

  always @(posedge clk) begin
    if (weA) memA[waA]    <= wdA;
    rdA <= memA[rd_addr];
  end
  always @(posedge clk) begin
    if (weB) memB[out_ptr] <= out_byte;
    rdB <= memB[rd_addr];
  end
  always @(posedge clk) rd_sel_q <= rd_sel;

  wire [7:0] rd_data = rd_sel_q ? rdB : rdA;
  wire signed [7:0] rd_s = rd_data;
  assign ro_data   = rd_data;
  assign done      = (st == S_DONE);
  assign class_idx = cls;

  // ------------------------------------------------------------ combinational helpers
  wire signed [11:0] oy_s   = {2'b00, oy};
  wire signed [11:0] ox_s   = {2'b00, ox};
  wire signed [11:0] ky_s   = {2'b00, ky};
  wire signed [11:0] kx_s   = {2'b00, kx};
  wire signed [11:0] str_s  = {2'b00, t_stride};
  wire signed [11:0] padh_s = {2'b00, t_padh};
  wire signed [11:0] padw_s = {2'b00, t_padw};
  wire signed [11:0] h_s    = {2'b00, t_h};
  wire signed [11:0] w_s    = {2'b00, t_w};

  wire [9:0] chan   = (t_typ == 2'd0) ? ic : oc;
  wire [9:0] ic_max = (t_typ == 2'd0) ? (t_cin - 10'd1) : 10'd0;
  wire [9:0] k_max  = t_k - 10'd1;

  wire [15:0] bsum = t_bbase + {6'd0, oc};

  wire signed [9:0] diff  = $signed({{2{rd_data[7]}}, rd_data}) - $signed({{2{t_zin[7]}}, t_zin});
  wire signed [7:0] w_eff = (t_typ == 2'd2) ? 8'sd1 : w_rd;

  wire [31:0] acc_abs = acc[31] ? (~acc[31:0] + 32'd1) : acc[31:0];
  wire [63:0] xabs    = -x64;

  // requant tail (combinational from registered q32 / exp_r)
  wire signed [63:0] q64    = q32;
  wire signed [63:0] mask   = (64'sd1 <<< exp_r) - 64'sd1;
  wire signed [63:0] rem    = q64 & mask;
  wire signed [63:0] thr    = (mask >>> 1) + (q64[63] ? 64'sd1 : 64'sd0);
  wire signed [63:0] res    = (q64 >>> exp_r) + ((rem > thr) ? 64'sd1 : 64'sd0);
  wire signed [63:0] zout_s = $signed({{56{t_zout[7]}}, t_zout});
  wire signed [63:0] yv     = res + zout_s;
  wire signed [63:0] cmin_s = $signed({{56{t_cmin[7]}}, t_cmin});
  wire signed [63:0] cmax_s = $signed({{56{t_cmax[7]}}, t_cmax});

  // ------------------------------------------------------------ main FSM
  always @(posedge clk) begin
    if (rst) begin
      st    <= S_IDLE;
      layer <= {`LW{1'b0}};
      wr_en <= 1'b0;
      rd_sel <= 1'b0;
    end else begin
      case (st)
        // ---------------------------------------------------------
        S_IDLE: begin
          if (start) begin
            layer <= {`LW{1'b0}};
            st    <= S_LINIT;
          end
        end
        // ---------------------------------------------------------
        S_LINIT: begin
          oy <= 10'd0; ox <= 10'd0; oc <= 10'd0;
          out_ptr <= {`AW{1'b0}};
          w_start <= t_wbase;
          rd_sel  <= layer[0];          // even layer reads A, odd reads B
          st      <= S_OINIT;
        end
        // ---------------------------------------------------------
        S_OINIT: begin
          bidx <= bsum[`BAW-1:0];
          ky <= 10'd0; kx <= 10'd0; ic <= 10'd0;
          w_ptr <= w_start;
          st <= S_B2;
        end
        S_B2: st <= S_B3;
        S_B3: begin
          acc    <= b_rd;
          mult_r <= m_rd;
          exp_r  <= e_rd;
          st     <= S_A1;
        end
        // ---------------------------------------------------------
        S_A1: begin
          iy_r <= oy_s * str_s + ky_s - padh_s;
          ix_r <= ox_s * str_s + kx_s - padw_s;
          st   <= S_A2;
        end
        S_A2: begin
          valid_r <= (iy_r >= 12'sd0) && (iy_r < h_s) && (ix_r >= 12'sd0) && (ix_r < w_s);
          row_r   <= iy_r * w_s + ix_r;
          st      <= S_A3;
        end
        S_A3: begin
          in_addr_r <= row_r[19:0] * {10'd0, t_cin} + {10'd0, chan};
          st <= valid_r ? S_RD : S_MAC;
        end
        S_RD: st <= S_MAC;
        S_MAC: begin
          if (valid_r) acc <= acc + diff * w_eff;
          w_ptr <= w_ptr + t_winc;
          if (ic != ic_max) begin
            ic <= ic + 10'd1;
            st <= S_A1;
          end else begin
            ic <= 10'd0;
            if (kx != k_max) begin
              kx <= kx + 10'd1;
              st <= S_A1;
            end else begin
              kx <= 10'd0;
              if (ky != k_max) begin
                ky <= ky + 10'd1;
                st <= S_A1;
              end else begin
                ky <= 10'd0;
                st <= S_MUL0;
              end
            end
          end
        end
        // ---------------------------------------------------------
        // requantize: shift-add 32x31 multiply, then gemmlowp rounding
        S_MUL0: begin
          acc_neg <= acc[31];
          m_sh    <= {32'd0, acc_abs};
          m_b     <= mult_r;
          m_p     <= 64'd0;
          m_cnt   <= 6'd0;
          st      <= S_MUL;
        end
        S_MUL: begin
          if (m_b[0]) m_p <= m_p + m_sh;
          m_sh  <= {m_sh[62:0], 1'b0};
          m_b   <= {1'b0, m_b[31:1]};
          m_cnt <= m_cnt + 6'd1;
          if (m_cnt == 6'd31) st <= S_RQ1;
        end
        S_RQ1: begin
          x64 <= (acc_neg ? -$signed(m_p) : $signed(m_p))
               + (acc_neg ? -64'sd1073741823 : 64'sd1073741824);
          st <= S_RQ2;
        end
        S_RQ2: begin
          if (x64[63]) q32 <= -$signed({1'b0, xabs[62:31]});
          else         q32 <= x64[62:31];
          st <= S_RQ3;
        end
        S_RQ3: begin
          out_byte <= (yv < cmin_s) ? t_cmin :
                      (yv > cmax_s) ? t_cmax : yv[7:0];
          wr_en <= 1'b1;
          st    <= S_STORE;
        end
        // ---------------------------------------------------------
        S_STORE: begin
          wr_en   <= 1'b0;
          out_ptr <= out_ptr + {{(`AW-1){1'b0}}, 1'b1};
          if (oc != t_cout - 10'd1) begin
            oc      <= oc + 10'd1;
            w_start <= w_start + t_wstride;
            st      <= S_OINIT;
          end else begin
            oc      <= 10'd0;
            w_start <= t_wbase;
            if (ox != t_ow - 10'd1) begin
              ox <= ox + 10'd1;
              st <= S_OINIT;
            end else begin
              ox <= 10'd0;
              if (oy != t_oh - 10'd1) begin
                oy <= oy + 10'd1;
                st <= S_OINIT;
              end else begin
                st <= S_LNEXT;
              end
            end
          end
        end
        // ---------------------------------------------------------
        S_LNEXT: begin
          if (layer == `NUM_LAYERS - 1) begin
            rd_sel <= ~layer[0];        // last layer's output buffer
            st     <= S_AG0;
          end else begin
            layer <= layer + {{(`LW-1){1'b0}}, 1'b1};
            st    <= S_LINIT;
          end
        end
        // ---------------------------------------------------------
        S_AG0: begin
          ag_i <= {`AW{1'b0}};
          best <= 8'sh80;
          cls  <= 8'd0;
          st   <= S_AG1;
        end
        S_AG1: st <= S_AG2;
        S_AG2: begin
          if (rd_s > best) begin
            best <= rd_s;
            cls  <= ag_i[7:0];
          end
          if (ag_i == `NUM_CLASSES - 1) st <= S_DONE;
          else begin
            ag_i <= ag_i + {{(`AW-1){1'b0}}, 1'b1};
            st   <= S_AG1;
          end
        end
        S_DONE: begin
          if (finish) st <= S_IDLE;
        end
        default: st <= S_IDLE;
      endcase
    end
  end

endmodule
