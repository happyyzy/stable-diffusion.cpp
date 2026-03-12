#!/usr/bin/env python3
"""Compute Phase5 metrics from GGML profile CSV + optional HTP prepack log."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List


PREPACK_RE = re.compile(
    r"GGML_HTP_PREPACK:\s+permute_calls=(\d+)\s+permute_bytes=(\d+)\s+permute_time=([0-9.]+)s\s+"
    r"repack_calls=(\d+)\s+repack_bytes=(\d+)\s+repack_time=([0-9.]+)s\s+total_time=([0-9.]+)s"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute copy/fallback/prepack metrics per step")
    p.add_argument("--op-csv", required=True, help="GGML_OP_PROFILE_CSV path")
    p.add_argument("--steps", type=int, required=True, help="number of denoise steps in the run")
    p.add_argument("--prepack-log", default="", help="stderr log containing GGML_HTP_PREPACK line")
    p.add_argument("--output-json", default="", help="write metrics json")
    return p.parse_args()


def to_int(row: Dict[str, str], key: str) -> int:
    v = row.get(key, "")
    if v is None or v == "":
        return 0
    try:
        return int(float(v))
    except ValueError:
        return 0


def load_op_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_prepack(log_path: Path) -> Dict[str, float]:
    out = {
        "permute_calls": 0,
        "permute_bytes": 0,
        "permute_time_s": 0.0,
        "repack_calls": 0,
        "repack_bytes": 0,
        "repack_time_s": 0.0,
        "total_time_s": 0.0,
    }
    if not log_path.exists():
        return out
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    m = PREPACK_RE.search(text)
    if not m:
        return out
    out["permute_calls"] = int(m.group(1))
    out["permute_bytes"] = int(m.group(2))
    out["permute_time_s"] = float(m.group(3))
    out["repack_calls"] = int(m.group(4))
    out["repack_bytes"] = int(m.group(5))
    out["repack_time_s"] = float(m.group(6))
    out["total_time_s"] = float(m.group(7))
    return out


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be > 0")

    rows = load_op_rows(Path(args.op_csv))
    total_cpu_us = sum(to_int(r, "cpu_us") for r in rows)
    total_copy_bytes = sum(
        to_int(r, "total_copy_bytes")
        if "total_copy_bytes" in r
        else to_int(r, "offload_copy_bytes") + to_int(r, "cpu_copy_bytes")
        for r in rows
    )

    top_cpu = sorted(
        ((r.get("op", ""), to_int(r, "cpu_us")) for r in rows),
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    prepack = parse_prepack(Path(args.prepack_log)) if args.prepack_log else parse_prepack(Path("/nonexistent"))

    metrics = {
        "inputs": {
            "op_csv": str(Path(args.op_csv).resolve()),
            "prepack_log": str(Path(args.prepack_log).resolve()) if args.prepack_log else "",
            "steps": args.steps,
        },
        "metrics": {
            "copy_bytes_per_step": total_copy_bytes / args.steps,
            "fallback_cpu_time_per_step_ms": (total_cpu_us / 1000.0) / args.steps,
            "repack_time_per_step_ms": (prepack["repack_time_s"] * 1000.0) / args.steps,
            "permute_time_per_step_ms": (prepack["permute_time_s"] * 1000.0) / args.steps,
            "prepack_total_time_per_step_ms": (prepack["total_time_s"] * 1000.0) / args.steps,
        },
        "totals": {
            "total_copy_bytes": total_copy_bytes,
            "total_cpu_us": total_cpu_us,
            "prepack": prepack,
        },
        "top_cpu_ops": [
            {"op": op, "cpu_us": cpu_us, "cpu_ms": cpu_us / 1000.0}
            for op, cpu_us in top_cpu
            if op
        ],
    }

    print(json.dumps(metrics, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"wrote: {Path(args.output_json).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
