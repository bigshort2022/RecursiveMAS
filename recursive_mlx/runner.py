"""Local-manifest execution; no implicit network or model acquisition."""

import importlib.metadata
import json
import os
import signal
import sys
from dataclasses import asdict
from pathlib import Path
from .bundle import EDGES, ROLES, sha256, validate_bundle
from .contracts import Budget
from .results import Receipts, RunLock, atomic_json, identity


def runtime_identity():
    from . import __version__

    root = Path(__file__).parent
    source = {p.name: sha256(p) for p in sorted(root.glob("*.py"))}
    from inference import prompts

    source["inference/prompts.py"] = sha256(prompts.__file__)
    versions = {}
    for name in ("mlx", "mlx-lm", "transformers", "tokenizers", "safetensors"):
        versions[name] = importlib.metadata.version(name)
    return {
        "port_version": __version__,
        "source": source,
        "dependencies": versions,
        "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32", "default"),
    }


def run(args, limits):
    data, root, bundle_id = validate_bundle(args.bundle)
    if (
        type(args.memory_budget_bytes) is not int
        or args.memory_budget_bytes < data["estimated_peak_bytes"]
    ):
        raise ValueError("declared bundle peak exceeds supplied memory budget")
    if args.input.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("input manifest exceeds 8 MiB limit")
    samples = json.loads(args.input.read_text())
    if not isinstance(samples, list) or not 1 <= len(samples) <= limits.max_samples:
        raise ValueError("input sample count must be within explicit max_samples")
    ids = set()
    for row in samples:
        if (
            not isinstance(row, dict)
            or set(row) != {"id", "question"}
            or not isinstance(row["id"], str)
            or not row["id"]
            or not isinstance(row["question"], str)
            or not row["question"]
            or len(row["question"]) > 100000
            or row["id"] in ids
        ):
            raise ValueError(
                "input requires unique string id and bounded nonempty question"
            )
        ids.add(row["id"])
    # Never put output inside verified model artifacts or overwrite the input/bundle.
    output = args.output.resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("results must be outside the immutable bundle")
    planned = {
        "status": "planned",
        "bundle_id": bundle_id,
        "samples": len(samples),
        "limits": asdict(limits),
        "estimated_peak_bytes": data["estimated_peak_bytes"],
        "memory_budget_bytes": args.memory_budget_bytes,
        "memory_enforcement": "cooperative_not_os_isolation",
        "model_quality": "unqualified",
    }
    if args.dry_run:
        return planned
    if not args.allow_unqualified:
        raise ValueError(
            "real-model backend is experimental; use --allow-unqualified after review"
        )
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        raise ValueError(
            "set MLX_ENABLE_TF32=0 before starting Python for the reference profile"
        )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    run_id = {
        "bundle_id": bundle_id,
        "input_hash": identity(samples),
        "limits": asdict(limits),
        "runtime": runtime_identity(),
        "generation": "greedy",
        "enable_thinking": False,
    }
    with RunLock(output):
        receipts = Receipts(output, run_id)
        pending = [row for row in samples if receipts.get(row["id"]) is None]
        resumed = len(samples) - len(pending)
        if not pending:
            return {
                "status": "completed",
                "completed": len(samples),
                "resumed": resumed,
                "run_hash": receipts.run_hash,
                "model_quality": "unqualified",
            }
        import mlx.core as mx
        from .backend import load_backend
        from .links import load_link
        from .engine import run_sequential

        cancelled = [False]
        old_handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        for s in old_handlers:
            signal.signal(s, lambda *_: cancelled.__setitem__(0, True))
        budget = Budget(limits, cancelled=lambda: cancelled[0])
        # MLX allocator limit is not an OS guarantee; stage checks and process ownership remain necessary.
        old_limit = mx.set_memory_limit(args.memory_budget_bytes)
        old_cache = mx.set_cache_limit(0)
        roles, inner, outer = {}, {}, {}
        try:

            def check_memory():
                budget.check()
                if mx.get_active_memory() > args.memory_budget_bytes:
                    raise MemoryError("observed MLX active bytes exceed budget")

            for role in ROLES:
                check_memory()
                roles[role] = load_backend(root / data["roles"][role]["model"])
                inner[role] = load_link(
                    root / data["roles"][role]["inner"], roles[role].width
                )
                mx.eval(inner[role].parameters())
                check_memory()
            for edge, (src, dst) in EDGES.items():
                outer[edge] = load_link(
                    root / data["outer"][edge], roles[src].width, roles[dst].width
                )
                mx.eval(outer[edge].parameters())
            check_memory()
            for row in pending:

                def emit(event):
                    check_memory()
                    print(
                        json.dumps({"schema": 1, "sample_id": row["id"], **event}),
                        file=sys.stderr,
                    )

                result = run_sequential(
                    roles, inner, outer, row["question"], budget, emit=emit
                )
                result["model_quality"] = "unqualified"
                receipts.put(row["id"], result)
            report = {
                "status": "completed",
                "completed": len(samples),
                "resumed": resumed,
                "run_hash": receipts.run_hash,
                "model_quality": "unqualified",
                "peak_mlx_bytes": mx.get_peak_memory(),
                "memory_enforcement": "cooperative_not_os_isolation",
            }
            atomic_json(output / "summary.json", report)
            return report
        finally:
            roles.clear()
            inner.clear()
            outer.clear()
            mx.clear_cache()
            mx.set_cache_limit(old_cache)
            mx.set_memory_limit(old_limit)
            for s, handler in old_handlers.items():
                signal.signal(s, handler)
