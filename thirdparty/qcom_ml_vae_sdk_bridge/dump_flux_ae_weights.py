#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
from safetensors import safe_open


def parse_args():
    parser = argparse.ArgumentParser(description="Dump Flux AE safetensors to CLML qfp16/qfp32 files.")
    parser.add_argument("--input", required=True, help="Path to Flux AE safetensors (e.g. ae.safetensors).")
    parser.add_argument("--output", required=True, help="Output decoder directory (e.g. weights/decoder).")
    parser.add_argument("--dtype", choices=["f16", "f32"], default="f16", help="Output dtype.")
    parser.add_argument(
        "--layout",
        choices=["auto", "legacy", "diffusers"],
        default="auto",
        help="Decoder key layout to normalize (auto detects by key patterns).",
    )
    parser.add_argument(
        "--all-keys",
        action="store_true",
        help="Dump all tensors. Default dumps only decoder.* tensors.",
    )
    parser.add_argument(
        "--include-post-quant-conv",
        action="store_true",
        help="Also dump post_quant_conv.{weight,bias} (required by Flux2 decoder path).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if an unknown decoder key pattern is encountered during normalization.",
    )
    return parser.parse_args()


def is_diffusers_decoder_key(key: str) -> bool:
    return (
        key.startswith("decoder.mid_block.")
        or key.startswith("decoder.up_blocks.")
        or key.startswith("decoder.conv_norm_out.")
    )


def normalize_diffusers_decoder_key(key: str) -> str | None:
    if key.startswith("decoder.conv_norm_out."):
        return key.replace("decoder.conv_norm_out.", "decoder.norm_out.", 1)

    if key.startswith("decoder.mid_block.attentions.0."):
        tail = key[len("decoder.mid_block.attentions.0.") :]
        tail_map = {
            "group_norm.weight": "norm.weight",
            "group_norm.bias": "norm.bias",
            "to_q.weight": "q.weight",
            "to_q.bias": "q.bias",
            "to_k.weight": "k.weight",
            "to_k.bias": "k.bias",
            "to_v.weight": "v.weight",
            "to_v.bias": "v.bias",
            "to_out.0.weight": "proj_out.weight",
            "to_out.0.bias": "proj_out.bias",
        }
        mapped = tail_map.get(tail)
        if mapped is None:
            return None
        return "decoder.mid.attn_1." + mapped

    m = re.match(r"^decoder\.mid_block\.resnets\.(\d+)\.(.+)$", key)
    if m:
        block_idx = int(m.group(1)) + 1
        return f"decoder.mid.block_{block_idx}.{m.group(2)}"

    parts = key.split(".")
    # decoder.up_blocks.{i}.resnets.{j}.xxx / decoder.up_blocks.{i}.upsamplers.0.conv.xxx
    if len(parts) >= 6 and parts[0] == "decoder" and parts[1] == "up_blocks":
        up_idx = 3 - int(parts[2])
        if parts[3] == "resnets" and len(parts) >= 7:
            block_idx = parts[4]
            tail = ".".join(parts[5:])
            if tail.startswith("conv_shortcut."):
                tail = tail.replace("conv_shortcut.", "nin_shortcut.", 1)
            return f"decoder.up.{up_idx}.block.{block_idx}.{tail}"
        if parts[3] == "upsamplers" and len(parts) >= 7 and parts[4] == "0" and parts[5] == "conv":
            tail = ".".join(parts[6:])
            return f"decoder.up.{up_idx}.upsample.conv.{tail}"

    return None


def normalize_decoder_key(key: str, layout: str) -> str | None:
    if layout == "legacy":
        return key
    if layout == "diffusers":
        return normalize_diffusers_decoder_key(key)
    # auto: convert only if key matches known diffusers layout markers
    if is_diffusers_decoder_key(key):
        return normalize_diffusers_decoder_key(key)
    return key


def should_dump_key(key: str, include_post_quant_conv: bool, all_keys: bool) -> bool:
    if all_keys:
        return True
    if key.startswith("decoder."):
        return True
    if include_post_quant_conv and key.startswith("post_quant_conv."):
        return True
    return False


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_dtype = np.float16 if args.dtype == "f16" else np.float32
    ext = ".qfp16" if args.dtype == "f16" else ".qfp32"

    dumped = 0
    skipped = 0
    unknown = 0
    normalized_seen = set()
    with safe_open(str(in_path), framework="numpy", device="cpu") as f:
        for key in f.keys():
            if not should_dump_key(key, args.include_post_quant_conv, args.all_keys):
                skipped += 1
                continue

            out_key = key
            if key.startswith("decoder."):
                mapped = normalize_decoder_key(key, args.layout)
                if mapped is None:
                    unknown += 1
                    if args.strict:
                        raise ValueError(f"unknown decoder key layout: {key}")
                    continue
                out_key = mapped

            elif key.startswith("post_quant_conv."):
                out_key = key

            if out_key in normalized_seen:
                raise ValueError(f"duplicate normalized key: {out_key}")
            normalized_seen.add(out_key)

            arr = f.get_tensor(key).astype(out_dtype, copy=False).reshape(-1, order="C")
            out_name = out_key.replace(".", "_") + ext
            out_path = out_dir / out_name
            with open(out_path, "wb") as wf:
                arr.tofile(wf)
            dumped += 1

    print(f"input={in_path}")
    print(f"output={out_dir}")
    print(f"dtype={args.dtype}")
    print(f"dumped={dumped} skipped={skipped}")
    if unknown > 0:
        print(f"unknown_decoder_keys={unknown}")


if __name__ == "__main__":
    main()
