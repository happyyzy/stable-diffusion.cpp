#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
from safetensors import safe_open


def parse_args():
    parser = argparse.ArgumentParser(description="Dump Flux AE safetensors to CLML qfp16/qfp32 files.")
    parser.add_argument("--input", required=True, help="Path to Flux AE safetensors (e.g. ae.safetensors).")
    parser.add_argument("--output", required=True, help="Output decoder directory (e.g. weights/decoder).")
    parser.add_argument("--dtype", choices=["f16", "f32"], default="f16", help="Output dtype.")
    parser.add_argument(
        "--all-keys",
        action="store_true",
        help="Dump all tensors. Default dumps only decoder.* tensors.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_dtype = np.float16 if args.dtype == "f16" else np.float32
    ext = ".qfp16" if args.dtype == "f16" else ".qfp32"

    dumped = 0
    skipped = 0
    with safe_open(str(in_path), framework="numpy", device="cpu") as f:
        for key in f.keys():
            if (not args.all_keys) and (not key.startswith("decoder.")):
                skipped += 1
                continue
            arr = f.get_tensor(key).astype(out_dtype, copy=False).reshape(-1, order="C")
            out_name = key.replace(".", "_") + ext
            out_path = out_dir / out_name
            with open(out_path, "wb") as wf:
                arr.tofile(wf)
            dumped += 1

    print(f"input={in_path}")
    print(f"output={out_dir}")
    print(f"dtype={args.dtype}")
    print(f"dumped={dumped} skipped={skipped}")


if __name__ == "__main__":
    main()
