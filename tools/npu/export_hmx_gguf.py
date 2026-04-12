#!/usr/bin/env python3
"""Generate or execute multi-variant HMX GGUF export jobs.

This is a workflow scaffold for stable-diffusion.cpp-npu.
It wraps the existing CLI convert mode and emits a reproducible manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class Variant:
    name: str
    out_type: str
    tensor_type_rules: str
    contract_id: str


@dataclass(frozen=True)
class TensorRouteRule:
    pattern: str
    preferred_backend: str
    op_family: str


ALL_VARIANTS: List[Variant] = [
    Variant(
        name="f16-hmx",
        out_type="f16",
        tensor_type_rules="",
        contract_id="sdcpp.hmx.v1.f16",
    ),
    Variant(
        name="q4_0_q8_0-hmx",
        out_type="q4_0",
        tensor_type_rules=(
            "attention\\.out\\.weight=q8_0,"
            "feed_forward\\.w2\\.weight=q8_0"
        ),
        contract_id="sdcpp.hmx.v1.q4q8",
    ),
    Variant(
        name="iq4_nl_q8_0-hmx",
        out_type="iq4_nl",
        tensor_type_rules=(
            "attention\\.out\\.weight=q8_0,"
            "feed_forward\\.w2\\.weight=q8_0"
        ),
        contract_id="sdcpp.hmx.v1.iq4q8",
    ),
    Variant(
        name="wf8-hmx",
        out_type="wf8_hmx",
        tensor_type_rules="",
        contract_id="sdcpp.hmx.v1.wf8",
    ),
]

VARIANT_BY_NAME = {variant.name: variant for variant in ALL_VARIANTS}

DEFAULT_VARIANT_NAMES = [
    "f16-hmx",
    "q4_0_q8_0-hmx",
    "iq4_nl_q8_0-hmx",
]


Z_IMAGE_ROUTE_RULES: List[TensorRouteRule] = [
    TensorRouteRule(r"\.attention\.qkv\.weight$", "HMX", "matmul"),
    TensorRouteRule(r"\.attention\.out\.weight$", "HMX", "matmul"),
    TensorRouteRule(r"\.feed_forward\.w1\.weight$", "HMX", "matmul"),
    TensorRouteRule(r"\.feed_forward\.w2\.weight$", "HMX", "matmul"),
    TensorRouteRule(r"\.feed_forward\.w3\.weight$", "HMX", "matmul"),
    TensorRouteRule(r"\.adaLN_modulation\.[01]\.weight$", "HMX", "matmul"),
    TensorRouteRule(r"\.x_embedder\.weight$", "HMX", "matmul"),
    TensorRouteRule(r"\.cap_embedder\.1\.weight$", "HMX", "matmul"),
]

MATMUL_LAYOUT_BY_QUANT = {
    "f16": "permute_32x32.v1",
    "wf8_hmx": "compact_f8.v1",
    "q4_0": "common_deq.v1",
    "q8_0": "common_deq.v1",
    "iq4_nl": "common_deq.v1",
}

FLASH_LAYOUT = "f32_q_f16_kv.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HMX GGUF export plan")
    parser.add_argument("--model", required=True, help="input model path")
    parser.add_argument("--output-dir", required=True, help="directory for exported gguf variants")
    parser.add_argument(
        "--cli-bin",
        default="./build-android-hex-step31/examples/cli/sdcpp",
        help="path to stable-diffusion.cpp CLI binary",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="output file prefix (default: model stem)",
    )
    parser.add_argument(
        "--manifest",
        default="export_manifest.json",
        help="manifest filename written under --output-dir",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute generated convert commands",
    )
    parser.add_argument(
        "--model-family",
        default="z_image",
        choices=["z_image"],
        help="routing template used for sidecar generation",
    )
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANT_NAMES),
        help=f"comma-separated variant names (available: {', '.join(VARIANT_BY_NAME.keys())})",
    )
    return parser.parse_args()


def shell_join(parts: List[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


def build_convert_cmd(cli_bin: Path, model_path: Path, out_path: Path, variant: Variant) -> List[str]:
    cmd = [
        str(cli_bin),
        "--mode",
        "convert",
        "--model",
        str(model_path),
        "--output",
        str(out_path),
        "--type",
        variant.out_type,
        "--convert-name",
    ]
    if variant.tensor_type_rules:
        cmd.extend(["--tensor-type-rules", variant.tensor_type_rules])
    return cmd


def parse_tensor_type_rules(tensor_type_rules: str) -> List[tuple[str, str]]:
    out: List[tuple[str, str]] = []
    if not tensor_type_rules:
        return out
    for item in tensor_type_rules.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        pattern, qtype = item.split("=", 1)
        pattern = pattern.strip()
        qtype = qtype.strip()
        if not pattern or not qtype:
            continue
        out.append((pattern, qtype))
    return out


def parse_variants(spec: str) -> List[Variant]:
    names = [item.strip() for item in spec.split(",") if item.strip()]
    if not names:
        raise ValueError("no variants selected")

    out: List[Variant] = []
    seen = set()
    for name in names:
        variant = VARIANT_BY_NAME.get(name)
        if variant is None:
            raise ValueError(
                f"unknown variant '{name}', available: {', '.join(VARIANT_BY_NAME.keys())}"
            )
        if name in seen:
            continue
        seen.add(name)
        out.append(variant)
    return out


def quant_for_tensor(name: str, default_qtype: str, rules: List[tuple[str, str]]) -> str:
    for pattern, qtype in rules:
        if re.search(pattern, name):
            return qtype
    return default_qtype


def load_safetensors_index_tensor_names(model_path: Path) -> List[str]:
    if model_path.suffix == ".json":
        index_path = model_path
    else:
        index_path = model_path.parent / "model.safetensors.index.json"
    if not index_path.exists():
        return []
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map", {})
        if isinstance(weight_map, dict):
            return sorted(str(k) for k in weight_map.keys())
    except Exception:
        return []
    return []


def build_sidecar_for_variant(model_path: Path, variant: Variant, output_path: Path, model_family: str) -> dict:
    if model_family != "z_image":
        raise ValueError(f"unsupported model_family: {model_family}")

    names = load_safetensors_index_tensor_names(model_path)
    quant_rules = parse_tensor_type_rules(variant.tensor_type_rules)
    entries = []

    if names:
        for name in names:
            for rule in Z_IMAGE_ROUTE_RULES:
                if not re.search(rule.pattern, name):
                    continue
                quant_type = quant_for_tensor(name, variant.out_type, quant_rules)
                entries.append(
                    {
                        "tensor_name": name,
                        "layout_contract_id": MATMUL_LAYOUT_BY_QUANT.get(quant_type, "unknown"),
                        "quant_type": quant_type,
                        "preferred_backend": rule.preferred_backend,
                        "requires_prepack": quant_type in {"f16", "wf8_hmx", "q4_0", "q8_0", "iq4_nl"},
                        "route_tag": rule.op_family,
                    }
                )
                break
    if not entries:
        for rule in Z_IMAGE_ROUTE_RULES:
            entries.append(
                {
                    "tensor_name_regex": rule.pattern,
                    "layout_contract_id": MATMUL_LAYOUT_BY_QUANT.get(variant.out_type, "unknown"),
                    "quant_type": variant.out_type,
                    "preferred_backend": rule.preferred_backend,
                    "requires_prepack": variant.out_type in {"f16", "wf8_hmx", "q4_0", "q8_0", "iq4_nl"},
                    "route_tag": rule.op_family,
                }
            )

    return {
        "contract_id": variant.contract_id,
        "model_path": str(model_path),
        "gguf_path": str(output_path),
        "layout_contracts": {
            "matmul": MATMUL_LAYOUT_BY_QUANT,
            "flash_attn_ext": FLASH_LAYOUT,
        },
        "tensor_routes": entries,
        "notes": {
            "source_tensor_names": "safetensors.index" if names else "regex_template",
            "z_image_refiner_scope": [
                "double_blocks.*",
                "context_refiner.*",
                "noise_refiner.*",
            ],
        },
    }


def main() -> int:
    args = parse_args()

    model_path = Path(args.model).resolve()
    output_dir = Path(args.output_dir).resolve()
    cli_bin = Path(args.cli_bin).resolve()

    if not model_path.exists():
        print(f"error: model not found: {model_path}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix.strip() or model_path.stem
    try:
        variants = parse_variants(args.variants)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    jobs = []
    for variant in variants:
        variant_dir = output_dir / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        out_path = variant_dir / f"{prefix}.gguf"
        sidecar_path = variant_dir / "model.npu_manifest.json"
        cmd = build_convert_cmd(cli_bin, model_path, out_path, variant)
        sidecar = build_sidecar_for_variant(model_path, variant, out_path, args.model_family)
        sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        jobs.append(
            {
                "variant": variant.name,
                "output": str(out_path),
                "sidecar": str(sidecar_path),
                "contract_id": variant.contract_id,
                "out_type": variant.out_type,
                "tensor_type_rules": variant.tensor_type_rules,
                "env": {
                    "SD_NPU_CONTRACT_ID": variant.contract_id,
                },
                "command": cmd,
                "command_shell": shell_join(cmd),
            }
        )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "output_dir": str(output_dir),
        "cli_bin": str(cli_bin),
        "model_family": args.model_family,
        "variants": [variant.name for variant in variants],
        "jobs": jobs,
    }

    manifest_path = output_dir / args.manifest
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote manifest: {manifest_path}")

    if not args.execute:
        print("dry-run only. pass --execute to run conversion jobs.")
        return 0

    if not cli_bin.exists():
        print(f"error: cli binary not found: {cli_bin}", file=sys.stderr)
        return 2

    for idx, job in enumerate(jobs, start=1):
        env = os.environ.copy()
        env.update(job["env"])
        print(f"[{idx}/{len(jobs)}] {job['variant']}: {job['command_shell']}")
        proc = subprocess.run(job["command"], env=env)
        if proc.returncode != 0:
            print(f"error: variant {job['variant']} failed with code {proc.returncode}", file=sys.stderr)
            return proc.returncode

    print("all export jobs completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
