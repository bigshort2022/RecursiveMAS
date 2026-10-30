"""Local-only, hash-verified bundle admission; no MLX or network imports."""

import hashlib
import json
import re
from pathlib import Path
from .results import identity

ROLES = ("planner", "critic", "solver")
EDGES = {
    "outer_12": ("planner", "critic"),
    "outer_23": ("critic", "solver"),
    "outer_31": ("solver", "planner"),
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_file(root, relative, digest):
    root = Path(root).resolve()
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError("artifact path escapes bundle")
    path = (root / p).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("artifact missing or outside bundle")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid sha256")
    if sha256(path) != digest:
        raise ValueError("artifact checksum mismatch: " + relative)
    return path


def validate_bundle(manifest):
    manifest = Path(manifest).resolve()
    if manifest.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("bundle manifest exceeds 8 MiB")
    data = json.loads(manifest.read_text())
    if not isinstance(data, dict):
        raise ValueError("bundle manifest must be an object")
    for key in ("roles", "outer", "files"):
        if not isinstance(data.get(key), dict):
            raise ValueError("bundle requires object: " + key)
    if data.get("schema") != 1 or data.get("style") != "sequential_light":
        raise ValueError("only schema 1 sequential_light bundles are supported")
    if set(data.get("roles", {})) != set(ROLES) or set(data.get("outer", {})) != set(
        EDGES
    ):
        raise ValueError(
            "bundle must contain three roles and three directed outer links"
        )
    root, inventory = manifest.parent, data.get("files", {})
    if not inventory:
        raise ValueError("bundle requires complete file checksums")
    paths = {
        name: checked_file(root, name, digest) for name, digest in inventory.items()
    }
    for role in ROLES:
        row = data["roles"][role]
        if not isinstance(row, dict) or not all(
            isinstance(row.get(k), str) and row[k]
            for k in ("model", "inner", "revision")
        ):
            raise ValueError("role requires model, inner and revision strings")
        model_dir = row["model"]
        local = (root / model_dir).resolve()
        if (
            Path(model_dir).is_absolute()
            or not local.is_relative_to(root)
            or not local.is_dir()
        ):
            raise ValueError("model directory must be local to bundle")
        for p in local.rglob("*"):
            if p.is_file() and p.relative_to(root).as_posix() not in inventory:
                raise ValueError("unhashed model file: " + str(p.relative_to(root)))
        cfg_path = local / "config.json"
        if cfg_path.relative_to(root).as_posix() not in paths:
            raise ValueError("missing model configuration")
        cfg = json.loads(cfg_path.read_text())
        if not isinstance(cfg, dict):
            raise ValueError("model config must be an object")
        expected = {"planner": "qwen3", "critic": "llama", "solver": "qwen2"}[role]
        if (
            cfg.get("model_type") != expected
            or cfg.get("quantization")
            or cfg.get("quantization_config")
        ):
            raise ValueError(
                "preview requires unquantized sequential-light family mapping"
            )
        if not isinstance(row.get("revision"), str) or not row["revision"]:
            raise ValueError("model revision required")
        if row["inner"] not in paths:
            raise ValueError("missing inner link")
        if not any(local.glob("*.safetensors")):
            raise ValueError("model safetensors missing")
    for name in EDGES:
        if not isinstance(data["outer"][name], str) or data["outer"][name] not in paths:
            raise ValueError("missing outer link")
    peak = data.get("estimated_peak_bytes")
    if type(peak) is not int or peak <= 0:
        raise ValueError(
            "positive estimated_peak_bytes required; it is not a measured guarantee"
        )
    return data, root, identity(data)
