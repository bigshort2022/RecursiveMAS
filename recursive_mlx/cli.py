"""No heavy imports until an explicit numerical command is selected."""

import argparse
import json
import os
import sys
from pathlib import Path
from .contracts import Limits, Cancelled


class UsageError(ValueError):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise UsageError(message)


def parser():
    p = Parser(
        description="Experimental RecursiveMAS MLX inference; local artifacts only."
    )
    sub = p.add_subparsers(dest="command")
    sub.add_parser("smoke", help="Run tiny random native MLX models, no downloads.")
    inspect = sub.add_parser(
        "inspect", help="Verify a local bundle; does not load models."
    )
    inspect.add_argument("bundle", type=Path)
    convert = sub.add_parser(
        "convert-link", help="Convert a local tensor-only .pt adapter."
    )
    convert.add_argument("source", type=Path)
    convert.add_argument("destination", type=Path)
    convert.add_argument("--kind", choices=["inner", "outer"], required=True)
    convert.add_argument("--in-dim", type=int, required=True)
    convert.add_argument("--out-dim", type=int, required=True)
    run = sub.add_parser(
        "run", help="Bounded greedy sequential-light math/choice inference."
    )
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument(
        "--input", type=Path, required=True, help="JSON array of id/question objects."
    )
    run.add_argument(
        "--output", type=Path, required=True, help="Private receipt directory."
    )
    run.add_argument("--memory-budget-bytes", type=int, required=True)
    run.add_argument("--rounds", type=int, default=1)
    run.add_argument("--latent-steps", type=int, default=4)
    run.add_argument("--max-tokens", type=int, default=32)
    run.add_argument("--max-context", type=int, default=2048)
    run.add_argument("--max-samples", type=int, default=1)
    run.add_argument("--timeout-s", type=float, default=120)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument(
        "--allow-unqualified",
        action="store_true",
        help="Acknowledge real-model quality/peak memory are not yet qualified.",
    )
    return p


def main(argv=None):
    try:
        args = parser().parse_args(argv)
        if args.command is None:
            out = {
                "schema": 1,
                "qualification": "experimental",
                "backend": "mlx",
                "style": "sequential_light",
                "sampling": "greedy",
                "model_quality": "unqualified",
                "training": "unsupported",
                "next": [
                    "python -m recursive_mlx --help",
                    "python -m recursive_mlx smoke",
                ],
            }
        elif args.command == "inspect":
            from .bundle import validate_bundle

            data, _, bundle_id = validate_bundle(args.bundle)
            out = {
                "bundle_id": bundle_id,
                "style": data["style"],
                "integrity": "verified",
                "estimated_peak_bytes": data["estimated_peak_bytes"],
                "qualification": "unqualified",
            }
        elif args.command == "smoke":
            os.environ.setdefault("MLX_ENABLE_TF32", "0")
            os.environ.setdefault("USE_TORCH", "0")
            from .engine import smoke

            out = smoke()
        elif args.command == "convert-link":
            from .conversion import convert_link

            out = convert_link(
                args.source, args.destination, args.kind, args.in_dim, args.out_dim
            )
        else:
            os.environ.setdefault("MLX_ENABLE_TF32", "0")
            os.environ.setdefault("USE_TORCH", "0")
            from .runner import run

            limits = Limits(
                rounds=args.rounds,
                latent_steps=args.latent_steps,
                max_tokens=args.max_tokens,
                max_context=args.max_context,
                max_samples=args.max_samples,
                timeout_s=args.timeout_s,
            )
            out = run(args, limits)
        print(json.dumps(out, allow_nan=False))
        return 0
    except UsageError as exc:
        print(json.dumps({"error": {"code": "usage", "message": str(exc)}}))
        return 2
    except (
        ValueError,
        OSError,
        RuntimeError,
        ImportError,
        TimeoutError,
        Cancelled,
        MemoryError,
    ) as exc:
        print(json.dumps({"error": {"code": type(exc).__name__, "message": str(exc)}}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
