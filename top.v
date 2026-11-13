// top.v - Nexys A7-100T: laptop sends an image over USB-UART, FPGA classifies it,
// shows the class number on the 7-segment display and replies over UART.
//
// Clock: the 100 MHz input is divided to 50 MHz by an MMCM; everything runs at 50 MHz
// (the core's longest path is ~12.6 ns, so 100 MHz does not meet timing).
// Protocol (8N1, 921600 baud):
//   laptop -> FPGA : 0xA5 0x5A, then IMG_BYTES bytes (int8, HWC order, RGB)
//   FPGA -> laptop : 0xA5, class_index, then NUM_CLASSES int8 logits
// Display: "CLA5      NN"  (NN = class index).  "--------" idle/busy, "EEEE" timeout.
// LEDs: 0 receiving, 1 computing, 2 result valid, 3 timeout error, 15:8 class index.
`timescale 1ns/1ps
`include "cnn_params.vh"

module top (
  input  wire        CLK100MHZ,
  input  wire        CPU_RESETN,
  input  wire        UART_TXD_IN,     // from PC (FPGA receives here)
  output wire        UART_RXD_OUT,    // to PC   (FPGA transmits here)
  output wire [15:0] LED,
  output wire [7:0]  AN,
  output wire [6:0]  SEG,
  output wire        DP
);
  localparam CLKS_PER_BIT = 54;       // 50 MHz / 921600 baud (0.5 % error)
  localparam TMO          = 25000000; // 0.5 s byte timeout inside an image

  // ---- 100 MHz -> 50 MHz (MMCM: VCO = 100 x 10 = 1000 MHz, out = 1000 / 20)
  wire clkfb_raw, clkfb, clk50_raw, locked, clk;
  MMCME2_BASE #(
    .BANDWIDTH("OPTIMIZED"),
    .CLKIN1_PERIOD(10.0),
    .CLKFBOUT_MULT_F(10.0),
    .DIVCLK_DIVIDE(1),
    .CLKOUT0_DIVIDE_F(20.0),
    .STARTUP_WAIT("FALSE")
  ) u_mmcm (
    .CLKIN1(CLK100MHZ), .CLKFBIN(clkfb), .CLKFBOUT(clkfb_raw),
    .CLKOUT0(clk50_raw), .LOCKED(locked), .PWRDWN(1'b0), .RST(1'b0)
  );
  BUFG u_bufg_fb  (.I(clkfb_raw),  .O(clkfb));
  BUFG u_bufg_clk (.I(clk50_raw),  .O(clk));

  // ---- reset synchroniser (button is active low; also hold reset until the MMCM locks)
  reg [2:0] rst_sr = 3'b111;
  always @(posedge clk) rst_sr <= {rst_sr[1:0], (~CPU_RESETN) | (~locked)};
  wire rst = rst_sr[2];

  // ---- UART
  wire       rx_valid;
  wire [7:0] rx_byte;
  uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (.clk(clk), .rx(UART_TXD_IN), .valid(rx_valid), .data(rx_byte));

  reg        tx_start = 1'b0;
  reg  [7:0] tx_data  = 8'd0;
  wire       tx_busy;
  uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (.clk(clk), .start(tx_start), .data(tx_data), .tx(UART_RXD_OUT), .busy(tx_busy));

  // ---- CNN core
  reg                start = 1'b0, finish = 1'b0, ld_we = 1'b0;
  reg  [`AW-1:0]     ld_addr = 0, ro_addr = 0;
  reg  [7:0]         ld_data = 8'd0;
  wire [7:0]         ro_data, class_idx;
  wire               core_done;
  cnn_core u_core (
    .clk(clk), .rst(rst), .start(start), .finish(finish),
    .ld_we(ld_we), .ld_addr(ld_addr), .ld_data(ld_data),
    .ro_addr(ro_addr), .ro_data(ro_data), .done(core_done), .class_idx(class_idx)
  );

  // ---- frame / reply controller
  localparam [3:0]
    F_H0 = 4'd0,  F_H1 = 4'd1,  F_DATA = 4'd2, F_START = 4'd3, F_RUN = 4'd4,
    F_T0 = 4'd5,  F_T1 = 4'd6,  F_L0 = 4'd7,   F_L1 = 4'd8,    F_L2 = 4'd9,
    F_FIN = 4'd10, F_TXGO = 4'd11, F_TXW1 = 4'd12, F_TXW2 = 4'd13;

  reg [3:0]      fs = F_H0, ret_st = F_H0;
  reg [`AW-1:0]  cnt = 0;
  reg [7:0]      lcnt = 8'd0;
  reg [25:0]     tmo = 26'd0;
  reg            err = 1'b0, res_valid = 1'b0;
  reg [7:0]      disp_class = 8'd0;

  always @(posedge clk) begin
    start    <= 1'b0;
    finish   <= 1'b0;
    ld_we    <= 1'b0;
    tx_start <= 1'b0;
    if (rst) begin
      fs <= F_H0; err <= 1'b0; res_valid <= 1'b0; tmo <= 26'd0;
    end else begin
      case (fs)
        F_H0: if (rx_valid && rx_byte == 8'hA5) fs <= F_H1;

        F_H1: if (rx_valid) begin
                if (rx_byte == 8'h5A) begin
                  cnt <= 0; tmo <= 26'd0; err <= 1'b0; res_valid <= 1'b0; fs <= F_DATA;
                end else if (rx_byte != 8'hA5) fs <= F_H0;
              end

        F_DATA: begin
          if (rx_valid) begin
            ld_we <= 1'b1; ld_addr <= cnt; ld_data <= rx_byte; tmo <= 26'd0;
            if (cnt == `IMG_BYTES - 1) fs <= F_START;
            else cnt <= cnt + 1'b1;
          end else if (tmo == TMO) begin
            err <= 1'b1; fs <= F_H0;
          end else tmo <= tmo + 26'd1;
        end

        F_START: begin start <= 1'b1; fs <= F_RUN; end

        F_RUN: if (core_done) begin
                 disp_class <= class_idx; res_valid <= 1'b1; fs <= F_T0;
               end

        F_T0: begin tx_data <= 8'hA5;      ret_st <= F_T1; fs <= F_TXGO; end
        F_T1: begin tx_data <= class_idx;  ret_st <= F_L0; lcnt <= 8'd0; fs <= F_TXGO; end

        F_L0: if (lcnt == `NUM_CLASSES) fs <= F_FIN;
              else begin ro_addr <= lcnt; fs <= F_L1; end
        F_L1: fs <= F_L2;
        F_L2: begin tx_data <= ro_data; lcnt <= lcnt + 8'd1; ret_st <= F_L0; fs <= F_TXGO; end

        F_FIN: begin finish <= 1'b1; fs <= F_H0; end

        // generic "send tx_data, then continue at ret_st"
        F_TXGO: if (!tx_busy) begin tx_start <= 1'b1; fs <= F_TXW1; end
        F_TXW1: fs <= F_TXW2;
        F_TXW2: if (!tx_busy) fs <= ret_st;

        default: fs <= F_H0;
      endcase
    end
  end

  // ---- LEDs
  assign LED[0]    = (fs == F_DATA);
  assign LED[1]    = (fs == F_START) || (fs == F_RUN);
  assign LED[2]    = res_valid;
  assign LED[3]    = err;
  assign LED[7:4]  = 4'd0;
  assign LED[15:8] = disp_class;

  // ---- 7-segment display
  wire [6:0] cls  = disp_class[6:0];
  wire [3:0] tens = cls / 7'd10;
  wire [3:0] ones = cls % 7'd10;
  wire [39:0] codes = err        ? {5'd15, 5'd15, 5'd15, 5'd15, 5'd11, 5'd11, 5'd11, 5'd11} :
                      res_valid  ? {5'd12, 5'd13, 5'd16, 5'd5, 5'd11, 5'd11, {1'b0, tens}, {1'b0, ones}} :
                                   {8{5'd10}};
  seg7_driver u_seg (.clk(clk), .codes(codes), .an(AN), .seg(SEG), .dp(DP));

endmodule
