#!/usr/bin/env python3
"""Generic HIP kernel benchmark with optional correctness validation.
Compiles .cpp with hipcc, loads via ctypes, validates against reference, benchmarks."""
import re, os, sys, subprocess, ctypes, argparse, importlib.util, torch

SUPPORTED_TYPES = {
    "float*": ctypes.c_void_p, "double*": ctypes.c_void_p, "int*": ctypes.c_void_p,
    "int": ctypes.c_int, "long": ctypes.c_long, "size_t": ctypes.c_size_t,
    "unsigned int": ctypes.c_uint,
}
DTYPE_MAP = {"float*": torch.float32, "double*": torch.float64, "int*": torch.int32}
INT_TYPES = {"int", "long", "size_t", "unsigned int"}

def parse_solve_signature(f):
    content = open(f).read()
    m = re.search(r'extern\s+"C"\s+void\s+solve\s*\(([\s\S]*?)\)\s*\{', content)
    if not m: raise ValueError(f'No extern "C" void solve(...) in {f}')
    raw = " ".join(re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", m.group(1))).split())
    params = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok: continue
        is_const = "const" in tok
        clean = re.sub(r"\s+", " ", tok.replace("const", "").strip())
        for key in sorted(SUPPORTED_TYPES, key=len, reverse=True):
            base = key.replace("*", r"\s*\*")
            pm = re.match(rf"({base})\s+(\w+)", clean)
            if pm: params.append((key, pm.group(2), is_const)); break
    return params

def detect_arch():
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        g = getattr(p, "gcnArchName", "")
        m = re.search(r"gfx\w+", g)
        if m: return m.group(0)
    return "gfx942"

def main():
    parser = argparse.ArgumentParser(description="HIP kernel benchmark")
    parser.add_argument("cpp_file"); parser.add_argument("--ref", default="")
    parser.add_argument("--warmup", type=int, default=10); parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--arch", default=""); parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-4); parser.add_argument("--rtol", type=float, default=1e-3)
    args, unknown = parser.parse_known_args()
    dims = {u[2:].split("=")[0]: int(u[2:].split("=")[1]) for u in unknown if u.startswith("--") and "=" in u}
    torch.cuda.set_device(args.gpu)
    arch = args.arch or detect_arch()
    params = parse_solve_signature(args.cpp_file)
    so = os.path.splitext(args.cpp_file)[0] + ".so"
    cmd = ["hipcc", "-shared", "-fPIC", f"--offload-arch={arch}", "-O3", "-o", so, args.cpp_file]
    print(f"[compile] {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0: print(r.stderr, file=sys.stderr); sys.exit(1)
    lib = ctypes.CDLL(so)
    int_vals = [dims[n] for t, n, _ in params if t in INT_TYPES]
    ptr_elems = int_vals[0] * int_vals[1] if len(int_vals) >= 2 else (int_vals[0] if int_vals else 1024*1024)
    ptr_elems = min(ptr_elems, 256*1024*1024)
    tensors, call_args, argtypes, outputs = {}, [], [], []
    for pt, pn, ic in params:
        if pt in DTYPE_MAP:
            t = torch.randn(ptr_elems, device="cuda", dtype=DTYPE_MAP[pt])
            tensors[pn] = t
            if not ic: outputs.append((pn, pt))
            call_args.append(ctypes.c_void_p(t.data_ptr())); argtypes.append(ctypes.c_void_p)
        elif pt in SUPPORTED_TYPES:
            call_args.append(SUPPORTED_TYPES[pt](dims[pn])); argtypes.append(SUPPORTED_TYPES[pt])
    lib.solve.restype = None; lib.solve.argtypes = argtypes
    if args.ref:
        spec = importlib.util.spec_from_file_location("_ref", args.ref)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        ref_t = {n: t.clone() for n, t in tensors.items()}
        ref_kw = {n: ref_t[n] if pt in DTYPE_MAP else dims[n] for pt, n, _ in params}
        lib.solve(*call_args); torch.cuda.synchronize()
        mod.reference(**ref_kw); torch.cuda.synchronize()
        ok = all(torch.allclose(tensors[n].float(), ref_t[n].float(), atol=args.atol, rtol=args.rtol) for n, _ in outputs)
        print(f"Validation: {'ALL PASS' if ok else 'FAILED'}")
        if not ok: sys.exit(1)
    for _ in range(args.warmup): lib.solve(*call_args)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(args.repeat): lib.solve(*call_args)
    e.record(); torch.cuda.synchronize()
    avg = s.elapsed_time(e) / args.repeat
    print(f"Average: {avg:.4f} ms | GPU: {torch.cuda.get_device_name(0)} | Arch: {arch}")

if __name__ == "__main__": main()
