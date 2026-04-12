#!/usr/bin/env python3
"""Phase0 baseline runner for stable-diffusion.cpp-npu.

Generates (and optionally executes) fixed 512/1024 single-step + 8-step commands.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class Case:
    name: str
    width: int
    height: int
    steps: int


CASES: List[Case] = [
    Case("hip_512_s1", 512, 512, 1),
    Case("hip_512_s8", 512, 512, 8),
    Case("hip_1024_s1", 1024, 1024, 1),
    Case("hip_1024_s8", 1024, 1024, 8),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate or run Phase0 HIP baseline cases")
    p.add_argument("--bin", required=True, help="path to sd-cli binary")
    p.add_argument("--model", required=True, help="GGUF model path")
    p.add_argument("--out-dir", required=True, help="output directory")
    p.add_argument("--prompt", default="a photo of an astronaut riding a horse")
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg-scale", type=float, default=7.0)
    p.add_argument("--sampling-method", default="euler")
    p.add_argument("--scheduler", default="discrete")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--execute", action="store_true", help="execute all cases")
    return p.parse_args()


def shell_join(args: List[str]) -> str:
    return " ".join(shlex.quote(a) for a in args)


def build_cmd(ns: argparse.Namespace, case: Case, image_out: Path) -> List[str]:
    cmd = [
        str(Path(ns.bin).resolve()),
        "--mode", "img_gen",
        "--model", str(Path(ns.model).resolve()),
        "-p", ns.prompt,
        "-n", ns.negative_prompt,
        "-W", str(case.width),
        "-H", str(case.height),
        "--steps", str(case.steps),
        "--sampling-method", ns.sampling_method,
        "--scheduler", ns.scheduler,
        "--cfg-scale", str(ns.cfg_scale),
        "--seed", str(ns.seed),
        "--threads", str(ns.threads),
        "-o", str(image_out),
    ]
    return cmd


def main() -> int:
    ns = parse_args()

    out_dir = Path(ns.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for case in CASES:
        image_out = out_dir / f"{case.name}.png"
        log_out = out_dir / f"{case.name}.log"
        cmd = build_cmd(ns, case, image_out)
        jobs.append(
            {
                "case": case.name,
                "width": case.width,
                "height": case.height,
                "steps": case.steps,
                "image_out": str(image_out),
                "log_out": str(log_out),
                "command": cmd,
                "command_shell": shell_join(cmd),
            }
        )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "phase0_hip_baseline",
        "seed": ns.seed,
        "sampling_method": ns.sampling_method,
        "scheduler": ns.scheduler,
        "cfg_scale": ns.cfg_scale,
        "threads": ns.threads,
        "jobs": jobs,
    }
    manifest_path = out_dir / "phase0_baseline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote: {manifest_path}")

    for i, job in enumerate(jobs, start=1):
        print(f"[{i}/{len(jobs)}] {job['command_shell']}")

    if not ns.execute:
        print("dry-run only. use --execute to run all cases.")
        return 0

    for i, job in enumerate(jobs, start=1):
        log_out = Path(job["log_out"])
        print(f"[run {i}/{len(jobs)}] {job['case']} -> {log_out}")
        with log_out.open("w", encoding="utf-8") as f:
            f.write(job["command_shell"] + "\n\n")
            proc = subprocess.run(job["command"], stdout=f, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            print(f"error: case {job['case']} failed ({proc.returncode})", file=sys.stderr)
            return proc.returncode

    print("phase0 baseline run completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
