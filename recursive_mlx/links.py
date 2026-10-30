"""Native RecursiveLink math; mirrors the MIT-licensed reference modeling.py."""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


class InnerLink(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj1 = nn.Linear(hidden_size, hidden_size)
        self.proj2 = nn.Linear(hidden_size, hidden_size)
        self.pre_ln = nn.LayerNorm(hidden_size, eps=1e-5)
        self.post_ln = nn.LayerNorm(hidden_size, eps=1e-5)

    def __call__(self, x):
        return self.post_ln(x + self.proj2(nn.gelu(self.proj1(self.pre_ln(x)))))


class OuterLink(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj1 = nn.Linear(in_dim, out_dim * 2)
        self.proj2 = nn.Linear(out_dim * 2, out_dim)
        self.ln_source = nn.LayerNorm(in_dim, eps=1e-5)
        self.ln_target = nn.LayerNorm(out_dim, eps=1e-5)
        self.residual_proj = nn.Linear(in_dim, out_dim)

    def __call__(self, x):
        return self.ln_target(
            self.proj2(nn.gelu(self.proj1(self.ln_source(x)))) + self.residual_proj(x)
        )


def validate_weights(module, weights):
    expected = dict(tree_flatten(module.parameters()))
    if set(expected) != set(weights):
        raise ValueError("link weight names do not match")
    for name, value in weights.items():
        if value.shape != expected[name].shape:
            raise ValueError("link weight shape mismatch: " + name)
        if (
            not mx.issubdtype(value.dtype, mx.floating)
            or not mx.all(mx.isfinite(value)).item()
        ):
            raise ValueError("link weights must be finite floating tensors")


def load_link(path, in_dim, out_dim=None):
    module = InnerLink(in_dim) if out_dim is None else OuterLink(in_dim, out_dim)
    weights = mx.load(str(path))
    validate_weights(module, weights)
    module.load_weights(list(weights.items()), strict=True)
    module.eval()
    return module


def apply_link(module, x, output_dtype=None):
    dtype = dict(tree_flatten(module.parameters()))["proj1.weight"].dtype
    y = module(x.astype(dtype))
    return y.astype(output_dtype) if output_dtype is not None else y
