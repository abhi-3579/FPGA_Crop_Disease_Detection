#!/usr/bin/env python3
"""
gen_hw.py  -  turn the verified export (out/layers.json + out/weights/*.mem) into
Verilog-ready files for the CNN core.

Usage:  python3 gen_hw.py [out_dir=out] [hw_dir=hw]

Writes into hw_dir:
  cnn_params.vh     `define constants (layer count, buffer sizes, ...)
  layer_table.v     combinational ROM of per-layer parameters
  weights_all.mem   all int8 weights, concatenated (hex, 1 byte / line)
  bias_all.mem      int32 bias per output channel (hex, 8 digits / line)
  mult_all.mem      int32 requant multiplier per output channel
  exp_all.mem       requant right-shift (rshift-31) per output channel
  tb_in.mem / tb_exp.mem   test image + expected logits (from out/golden/in_00, out_00) if present
"""
import json, math, os, shutil, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
HW = sys.argv[2] if len(sys.argv) > 2 else "hw"
os.makedirs(HW, exist_ok=True)

meta = json.load(open(f"{OUT}/layers.json"))
layers = [l for l in meta["layers"] if l["op"] != "DELEGATE"]   # ignore XNNPACK pseudo-op


def read_mem(fname, bits):
    vals = []
    for tok in open(f"{OUT}/weights/{fname}").read().split():
        v = int(tok, 16)
        if v >= 1 << (bits - 1):
            v -= 1 << bits
        vals.append(v)
    return vals


def requant_params(M):
    m, e = math.frexp(M)
    mult, rshift = int(round(m * 2**31)), 31 - e
    if mult == 2**31:
        mult //= 2
        rshift -= 1
    return mult, rshift


def s8(v):                      # 8-bit literal (two's complement) for a signed reg
    return "8'h%02x" % (v & 0xFF)


TYPE = {"CONV_2D": 0, "DEPTHWISE_CONV_2D": 1, "MEAN": 2, "FULLY_CONNECTED": 0}

W_ALL, B_ALL, M_ALL, E_ALL = [], [], [], []
rows = []
act_max = None

for n, L in enumerate(layers):
    op = L["op"]
    if op not in TYPE:
        sys.exit(f"Unsupported op {op} in layer {n}; tell Claude which op this is")
    ish, osh = L["in_shape"], L["out_shape"]
    if len(ish) == 4:
        H, W, C = ish[1], ish[2], ish[3]
    else:
        H, W, C = 1, 1, ish[1]
    if len(osh) == 4:
        OH, OW, CO = osh[1], osh[2], osh[3]
    else:
        OH, OW, CO = 1, 1, osh[1]
    if act_max is None:
        act_max = H * W * C
    act_max = max(act_max, OH * OW * CO)

    if op == "MEAN":
        assert H == W, "MEAN over non-square map not supported"
        K, stride, cin = H, 1, C
        wstride, winc = 0, 0
        n_pool = L["pool_divisor"]
        mult, rsh = requant_params(L["in_scale"] / L["out_scale"] / n_pool)
        mults, rshs = [mult] * CO, [rsh] * CO
        biases = [0] * CO
        cmin, cmax = -128, 127
        wbase = len(W_ALL)                         # unused
    else:
        ws = L["w_shape"]
        w = read_mem(L["w_file"], 8)
        b = read_mem(L["b_file"], 32)
        assert len(w) == math.prod(ws) and len(b) == CO, f"size mismatch layer {n}"
        if op == "DEPTHWISE_CONV_2D":
            K, cin = ws[1], C
            assert CO == C and ws[3] == C and ws[0] == 1
            wstride, winc = 1, C
        elif op == "CONV_2D":
            K, cin = ws[1], ws[3]
            assert cin == C and ws[0] == CO
            wstride, winc = K * K * cin, 1
        else:                                       # FULLY_CONNECTED, weights [out, in]
            K, cin = 1, ws[1]
            assert cin == C and ws[0] == CO
            wstride, winc = cin, 1
        assert H % OH == 0 or op == "FULLY_CONNECTED"
        stride = H // OH if op != "FULLY_CONNECTED" else 1
        wbase = len(W_ALL)
        W_ALL += w
        biases = b
        mults, rshs = L["mult"], L["rshift"]
        cmin, cmax = L["clamp_min"], L["clamp_max"]

    padh = max((OH - 1) * stride + K - H, 0) // 2
    padw = max((OW - 1) * stride + K - W, 0) // 2
    exps = [r - 31 for r in rshs]
    assert all(0 <= e <= 62 for e in exps), f"layer {n}: requant exponent out of range {min(exps)}..{max(exps)}"
    assert all(2**30 <= m < 2**31 for m in mults), f"layer {n}: multiplier out of range"
    assert max(H, W, cin, OH, OW, CO, K) < 1024
    bbase = len(B_ALL)
    B_ALL += biases
    M_ALL += mults
    E_ALL += exps
    rows.append(dict(typ=TYPE[op], h=H, w=W, cin=cin, oh=OH, ow=OW, cout=CO, k=K,
                     stride=stride, padh=padh, padw=padw, wbase=wbase, wstride=wstride,
                     winc=winc, bbase=bbase, zin=L["in_zp"], zout=L["out_zp"],
                     cmin=cmin, cmax=cmax))

NL = len(layers)
NCLS = rows[-1]["cout"]
IMG = rows[0]["h"] * rows[0]["w"] * rows[0]["cin"]
AW = max(1, (act_max - 1).bit_length())
LW = max(1, (NL - 1).bit_length())

# ---- memory files ----
open(f"{HW}/weights_all.mem", "w").write("\n".join("%02x" % (v & 0xFF) for v in W_ALL) + "\n")
open(f"{HW}/bias_all.mem", "w").write("\n".join("%08x" % (v & 0xFFFFFFFF) for v in B_ALL) + "\n")
open(f"{HW}/mult_all.mem", "w").write("\n".join("%08x" % v for v in M_ALL) + "\n")
open(f"{HW}/exp_all.mem", "w").write("\n".join("%02x" % v for v in E_ALL) + "\n")

# ---- header ----
open(f"{HW}/cnn_params.vh", "w").write(f"""// AUTO-GENERATED by gen_hw.py - do not edit
`define NUM_LAYERS {NL}
`define LW {LW}
`define NUM_CLASSES {NCLS}
`define IMG_BYTES {IMG}
`define ACT_DEPTH {act_max}
`define AW {AW}
`define WN {len(W_ALL)}
`define WAW {max(1, (len(W_ALL) - 1).bit_length())}
`define BN {len(B_ALL)}
`define BAW {max(1, (len(B_ALL) - 1).bit_length())}
""")

# ---- layer table ----
lines = ["// AUTO-GENERATED by gen_hw.py - do not edit",
         '`include "cnn_params.vh"',
         "module layer_table (",
         "  input  wire [`LW-1:0] idx,",
         "  output reg  [1:0]  typ,          // 0 = conv/fc, 1 = depthwise, 2 = mean",
         "  output reg  [9:0]  h, w, cin, oh, ow, cout, k, stride, padh, padw,",
         "  output reg  [19:0] wbase, wstride, winc,",
         "  output reg  [15:0] bbase,",
         "  output reg  [7:0]  zin, zout, cmin, cmax   // two's complement",
         ");",
         "always @* begin",
         "  case (idx)"]
for i, r in enumerate(rows):
    lines.append(f"    {LW}'d{i}: begin")
    lines.append(f"      typ=2'd{r['typ']}; h=10'd{r['h']}; w=10'd{r['w']}; cin=10'd{r['cin']}; "
                 f"oh=10'd{r['oh']}; ow=10'd{r['ow']}; cout=10'd{r['cout']}; k=10'd{r['k']};")
    lines.append(f"      stride=10'd{r['stride']}; padh=10'd{r['padh']}; padw=10'd{r['padw']};")
    lines.append(f"      wbase=20'd{r['wbase']}; wstride=20'd{r['wstride']}; winc=20'd{r['winc']}; bbase=16'd{r['bbase']};")
    lines.append(f"      zin={s8(r['zin'])}; zout={s8(r['zout'])}; cmin={s8(r['cmin'])}; cmax={s8(r['cmax'])};")
    lines.append("    end")
lines += ["    default: begin",
          "      typ=2'd0; h=10'd0; w=10'd0; cin=10'd0; oh=10'd0; ow=10'd0; cout=10'd0; k=10'd0;",
          "      stride=10'd0; padh=10'd0; padw=10'd0; wbase=20'd0; wstride=20'd0; winc=20'd0; bbase=16'd0;",
          "      zin=8'h00; zout=8'h00; cmin=8'h00; cmax=8'h00;",
          "    end",
          "  endcase",
          "end",
          "endmodule"]
open(f"{HW}/layer_table.v", "w").write("\n".join(lines) + "\n")

# ---- testbench vectors (first golden image) ----
for src, dst in (("in_00.mem", "tb_in.mem"), ("out_00.mem", "tb_exp.mem")):
    p = f"{OUT}/golden/{src}"
    if os.path.exists(p):
        shutil.copy(p, f"{HW}/{dst}")
    else:
        print(f"note: {p} not found; run golden_check.py first to create testbench vectors")

print(f"layers={NL} classes={NCLS} img_bytes={IMG} act_depth={act_max} (AW={AW}) "
      f"weights={len(W_ALL)} B per-channel entries={len(B_ALL)}")
print("Generated into", HW)
