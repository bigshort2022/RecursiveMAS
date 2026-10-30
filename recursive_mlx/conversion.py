"""Explicit local tensor-only conversion. PyTorch is a conversion-only dependency."""

import os
import pickle
import shutil
import tempfile
from pathlib import Path
from .bundle import sha256
from .results import atomic_json


def convert_link(source, destination, kind, in_dim, out_dim):
    import torch
    import mlx.core as mx
    from .links import InnerLink, OuterLink, validate_weights

    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError("destination already exists")
    if kind not in ("inner", "outer") or min(in_dim, out_dim) <= 0:
        raise ValueError("invalid link kind or dimensions")
    if kind == "inner" and in_dim != out_dim:
        raise ValueError("inner link dimensions must match")
    source_hash = sha256(source)
    try:
        state = torch.load(source, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError) as exc:
        # Never retry with unrestricted pickle or add deserialization allowlists.
        raise ValueError(
            "source must be a supported tensor-only state dictionary"
        ) from exc
    if not isinstance(state, dict) or not all(
        isinstance(v, torch.Tensor) for v in state.values()
    ):
        raise ValueError("source must be a tensor-only state dictionary")
    weights = {}
    for name, tensor in state.items():
        if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("unsupported source tensor dtype")
        if not torch.isfinite(tensor).all():
            raise ValueError("non-finite source tensor")
        # Preserve bf16 via float32 import then explicitly cast; never silently change stored dtype.
        dtype = {
            torch.float32: mx.float32,
            torch.float16: mx.float16,
            torch.bfloat16: mx.bfloat16,
        }[tensor.dtype]
        weights[name] = mx.array(tensor.float().numpy()).astype(dtype)
    if sha256(source) != source_hash:
        raise ValueError("source changed during conversion")
    module = InnerLink(in_dim) if kind == "inner" else OuterLink(in_dim, out_dim)
    validate_weights(module, weights)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".conversion-", dir=destination.parent))
    try:
        mx.save_safetensors(str(stage / "weights.safetensors"), weights)
        record = {
            "schema": 1,
            "kind": kind,
            "in_dim": in_dim,
            "out_dim": out_dim,
            "source_sha256": source_hash,
            "sha256": sha256(stage / "weights.safetensors"),
            "epsilon": 1e-5,
            "gelu": "exact",
            "layout": "out_in",
            "dtypes": {k: str(v.dtype) for k, v in weights.items()},
        }
        atomic_json(stage / "conversion.json", record)
        # Never replace an existing destination, including an empty directory.
        destination.mkdir()
        try:
            # The manifest is the publication marker and is always moved last.
            for name in ("weights.safetensors", "conversion.json"):
                os.replace(stage / name, destination / name)
            parent_fd = os.open(destination, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except BaseException:
            shutil.rmtree(destination)
            raise
        return record
    finally:
        shutil.rmtree(stage)
