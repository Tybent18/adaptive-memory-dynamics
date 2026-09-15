"""Command-line entry point."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .experiment import run_suite
from .visualization import create_charts


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Run Adaptive Memory Dynamics experiments")
    command.add_argument("--output", type=Path)
    command.add_argument("--seeds", default="7,21,42,84,101")
    command.add_argument("--epochs", type=int, default=24)
    command.add_argument("--continual-epochs", type=int, default=16)
    command.add_argument("--quick", action="store_true", help="two-seed CI/development run")
    return command


def main() -> int:
    args = parser().parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("results/runs") / stamp
    seeds = [7, 42] if args.quick else [int(seed) for seed in args.seeds.split(",")]
    epochs = min(args.epochs, 6) if args.quick else args.epochs
    continual = min(args.continual_epochs, 4) if args.quick else args.continual_epochs
    histories, summaries = run_suite(output, seeds, epochs, continual, print_progress)
    create_charts(histories, summaries, output)
    print(f"\nEvidence bundle written to {output}")
    return 0


def print_progress(message: str, progress: float) -> None:
    print(f"[{progress:6.1%}] {message}")


if __name__ == "__main__":
    raise SystemExit(main())
