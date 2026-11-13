#!/usr/bin/env python3
"""
verify_twin.py - checks the generated hardware files WITHOUT Vivado.

It re-implements, step for step, the algorithm of cnn_core.v (same loop order, same
incremental weight pointer, same padding test, same shift-add multiplier and rounding)
using ONLY the generated files (hw/layer_table.v, hw/*.mem), and compares the logits
with out/golden/out_XX.mem.   If this passes, the tables and the algorithm are right,
and any remaining mismatch in simulation is a Verilog coding/timing bug.

Usage: python3 verify_twin.py [out_dir=out] [hw_dir=hw] [num_images=3]
"""
import re, sys, glob

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
HW = sys.argv[2] if len(sys.argv) > 2 else "hw"
NIMG = int(sys.argv[3]) if len(sys.argv) > 3 else 3


def hexlist(path, bits):
    v = []
    for t in open(path).read().split():
        x = int(t, 16)
        v.append(x - (1 << bits) if x >= 1 << (bits - 1) else x)
    return v


def macros(path):
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"`define (\w+) (\d+)", open(path).read())}


def parse_table(path):
    txt = open(path).read()
    table = {}
    for m in re.finditer(r"\d+'d(\d+): begin(.*?)\n    end", txt, re.S):
        idx = int(m.group(1))
        f = {}
        for name, width, base, val in re.findall(r"(\w+)=(\d+)'([dh])([0-9a-fA-F]+);", m.group(2)):
            n = int(val, 16 if base == "h" else 10)
            if name in ("zin", "zout", "cmin", "cmax") and n >= 128:
                n -= 256
            f[name] = n
        table[idx] = f
    return table


P = macros(f"{HW}/cnn_params.vh")
TBL = parse_table(f"{HW}/layer_table.v")
WROM = hexlist(f"{HW}/weights_all.mem", 8)
BROM = hexlist(f"{HW}/bias_all.mem", 32)
MROM = [int(t, 16) for t in open(f"{HW}/mult_all.mem").read().split()]
EROM = [int(t, 16) for t in open(f"{HW}/exp_all.mem").read().split()]
NL, NCLS, DEPTH = P["NUM_LAYERS"], P["NUM_CLASSES"], P["ACT_DEPTH"]


def requant_like_verilog(acc, mult, ex, zout, cmin, cmax):
    neg = acc < 0
    a = (-acc) & 0xFFFFFFFF if neg else acc                 # 32-bit magnitude
    prod, sh, b = 0, a, mult                                # shift-add multiplier, 32 iterations
    for _ in range(32):
        if b & 1:
            prod += sh
        sh <<= 1
        b >>= 1
    ab = -prod if neg else prod
    x = ab + (1 - (1 << 30) if neg else (1 << 30))
    q = (x >> 31) if x >= 0 else -((-x) >> 31)
    mask = (1 << ex) - 1
    rem = q & mask
    thr = (mask >> 1) + (1 if q < 0 else 0)
    res = (q >> ex) + (1 if rem > thr else 0)
    y = res + zout
    return min(max(y, cmin), cmax)


def run(img):
    bufs = [list(img) + [0] * (DEPTH - len(img)), [0] * DEPTH]
    for layer in range(NL):
        t = TBL[layer]
        src, dst = bufs[layer % 2], bufs[1 - layer % 2]
        w_start, out_ptr = t["wbase"], 0
        ic_max = t["cin"] - 1 if t["typ"] == 0 else 0
        for oy in range(t["oh"]):
            for ox in range(t["ow"]):
                for oc in range(t["cout"]):
                    bidx = t["bbase"] + oc
                    acc, w_ptr = BROM[bidx], w_start
                    for ky in range(t["k"]):
                        for kx in range(t["k"]):
                            for ic in range(ic_max + 1):
                                iy = oy * t["stride"] + ky - t["padh"]
                                ix = ox * t["stride"] + kx - t["padw"]
                                valid = 0 <= iy < t["h"] and 0 <= ix < t["w"]
                                chan = ic if t["typ"] == 0 else oc
                                if valid:
                                    x = src[(iy * t["w"] + ix) * t["cin"] + chan]
                                    wv = 1 if t["typ"] == 2 else WROM[w_ptr]
                                    acc += (x - t["zin"]) * wv
                                w_ptr += t["winc"]
                    assert -2**31 <= acc < 2**31, "accumulator overflow"
                    dst[out_ptr] = requant_like_verilog(acc, MROM[bidx], EROM[bidx],
                                                        t["zout"], t["cmin"], t["cmax"])
                    out_ptr += 1
                    w_start = w_start + t["wstride"] if oc != t["cout"] - 1 else t["wbase"]
    final = bufs[1 - (NL - 1) % 2][:NCLS]
    best, cls = -128, 0
    for i, v in enumerate(final):
        if v > best:
            best, cls = v, i
    return final, cls


ins = sorted(glob.glob(f"{OUT}/golden/in_*.mem"))[:NIMG]
if not ins:
    sys.exit("No out/golden/in_XX.mem files: run golden_check.py first")
ok = 0
for p in ins:
    exp = hexlist(p.replace("in_", "out_"), 8)
    got, cls = run(hexlist(p, 8))
    same = got == exp
    ok += same
    print(f"{p.split('/')[-1]}: logits {'MATCH' if same else 'DIFFER'}  class={cls}"
          f"{'' if same else '  max|diff|=' + str(max(abs(a - b) for a, b in zip(got, exp)))}")
print("TWIN PASS" if ok == len(ins) else "TWIN FAIL")
