"""Numerical failure must not become a successful generated answer."""

import unittest

import mlx.core as mx

from recursive_mlx.backend import tiny_backend
from recursive_mlx.contracts import Budget, Limits
from recursive_mlx.engine import latent_rollout, run_sequential
from recursive_mlx.links import InnerLink, OuterLink, apply_link


class InferenceSafetyTests(unittest.TestCase):
    def test_nonfinite_decoder_states_rejected(self):
        backend = tiny_backend()
        backend.model.model.embed_tokens.weight = mx.full_like(
            backend.model.model.embed_tokens.weight, float("nan")
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            latent_rollout(
                backend, InnerLink(8), backend.embed([1, 2]), 1, Budget(Limits())
            )

    def test_nonfinite_link_cast_rejected(self):
        link = InnerLink(8)
        # Finite FP32 link weights can still overflow a destination embedding dtype.
        link.post_ln.bias = mx.full((8,), 1e10)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            apply_link(link, mx.zeros((1, 1, 8)), mx.float16)

    def test_nonfinite_solver_logits_rejected(self):
        roles = {role: tiny_backend() for role in ("planner", "critic", "solver")}
        roles["solver"].model.model.embed_tokens.weight = mx.full_like(
            roles["solver"].model.model.embed_tokens.weight, float("nan")
        )
        inner = {role: InnerLink(8) for role in roles}
        outer = {edge: OuterLink(8, 8) for edge in ("outer_12", "outer_23", "outer_31")}

        def prompt(backend, role, question, incoming):
            return backend.embed([1, 2])

        with self.assertRaisesRegex(ValueError, "non-finite"):
            run_sequential(
                roles,
                inner,
                outer,
                "fixture",
                Budget(Limits(latent_steps=1, max_tokens=1)),
                build_prompt=prompt,
            )


if __name__ == "__main__":
    unittest.main()
