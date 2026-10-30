import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np


class NativeTests(unittest.TestCase):
    def test_links_against_torch(self):
        import torch
        import torch.nn as nn
        from recursive_mlx.links import InnerLink, OuterLink
        from mlx.utils import tree_flatten

        torch.set_num_threads(1)
        for link in [InnerLink(4), OuterLink(4, 6)]:
            weights = dict(tree_flatten(link.parameters()))
            x = mx.array(
                np.random.default_rng(7).normal(size=(1, 3, 4)).astype("float32")
            )
            t = torch.tensor(np.array(x))
            w = {k: torch.tensor(np.array(v)) for k, v in weights.items()}

            def linear(a, n):
                return nn.functional.linear(a, w[n + ".weight"], w[n + ".bias"])

            def norm(a, n):
                return nn.functional.layer_norm(
                    a, (a.shape[-1],), w[n + ".weight"], w[n + ".bias"], 1e-5
                )

            if isinstance(link, InnerLink):
                y = norm(t, "pre_ln")
                y = norm(
                    t + linear(nn.functional.gelu(linear(y, "proj1")), "proj2"),
                    "post_ln",
                )
            else:
                y = norm(t, "ln_source")
                y = norm(
                    linear(nn.functional.gelu(linear(y, "proj1")), "proj2")
                    + linear(t, "residual_proj"),
                    "ln_target",
                )
            np.testing.assert_allclose(
                np.array(link(x)), y.numpy(), atol=1e-5, rtol=1e-4
            )

    def test_required_families_native_shapes(self):
        from recursive_mlx.backend import tiny_backend

        for family in ("qwen3", "qwen2", "llama"):
            with self.subTest(family=family):
                b = tiny_backend(family, hidden_size=8)
                e = b.embed([1, 2, 3])
                self.assertEqual(e.shape, (1, 3, 8))
                h = b.hidden(e)
                self.assertEqual(h.shape, e.shape)
                self.assertEqual(b.logits(h).shape, (1, 3, 32))
                self.assertTrue(bool(mx.all(mx.isfinite(h)).item()))

    def test_decoder_parity_against_torch(self):
        import torch
        import transformers
        from mlx.utils import tree_flatten
        from recursive_mlx.backend import tiny_backend

        torch.set_num_threads(1)
        for family, cls in (("qwen3", "Qwen3"), ("qwen2", "Qwen2"), ("llama", "Llama")):
            with self.subTest(family=family):
                b = tiny_backend(family, 8)
                cfg = getattr(transformers, cls + "Config")(
                    hidden_size=8,
                    num_hidden_layers=1,
                    intermediate_size=16,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=4,
                    vocab_size=32,
                    max_position_embeddings=128,
                    rope_theta=10000.0,
                    rms_norm_eps=1e-6,
                    tie_word_embeddings=True,
                )
                cfg._attn_implementation = "eager"
                ref = getattr(transformers, cls + "ForCausalLM")(cfg).float().eval()
                weights = {
                    k: torch.tensor(np.array(v))
                    for k, v in tree_flatten(b.model.parameters())
                }
                weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
                ref.load_state_dict(weights, strict=True)
                e = b.embed([1, 2, 3])
                with torch.no_grad():
                    expected = ref.model(
                        inputs_embeds=torch.tensor(np.array(e)), use_cache=False
                    ).last_hidden_state
                np.testing.assert_allclose(
                    np.array(b.hidden(e)), expected.numpy(), atol=1e-5, rtol=1e-4
                )

    def test_rollout_matches_reference_and_bounds(self):
        from recursive_mlx.backend import tiny_backend
        from recursive_mlx.links import InnerLink
        from recursive_mlx.engine import latent_rollout
        from recursive_mlx.contracts import Budget, Limits

        b, link = tiny_backend("qwen3", 8), InnerLink(8)
        e = b.embed([1, 2])
        got = latent_rollout(b, link, e, 3, Budget(Limits(max_context=32)))
        current, expected = e, []
        for _ in range(3):
            last = b.hidden(current)[:, -1:, :]
            mx.eval(last)
            expected.append(last)
            current = mx.concatenate([current, link(last)], axis=1)
        np.testing.assert_allclose(
            np.array(got),
            np.array(mx.concatenate(expected, axis=1)),
            atol=1e-5,
            rtol=1e-4,
        )
        with self.assertRaises(ValueError):
            latent_rollout(b, link, e, 3, Budget(Limits(max_context=3)))

    def test_conversion_strict_and_local(self):
        import torch
        from mlx.utils import tree_flatten
        from recursive_mlx.links import InnerLink
        from recursive_mlx.conversion import convert_link

        with tempfile.TemporaryDirectory() as d:
            source, dest = Path(d) / "adapter.pt", Path(d) / "converted"
            weights = {
                k: torch.tensor(np.array(v))
                for k, v in tree_flatten(InnerLink(4).parameters())
            }
            torch.save(weights, source)
            manifest = convert_link(source, dest, "inner", 4, 4)
            self.assertEqual(manifest["kind"], "inner")
            self.assertTrue((dest / "weights.safetensors").is_file())
            with self.assertRaises(FileExistsError):
                convert_link(source, dest, "inner", 4, 4)
            weights["proj1.weight"] = torch.ones(2, 2)
            torch.save(weights, source)
            with self.assertRaises(ValueError):
                convert_link(source, Path(d) / "bad", "inner", 4, 4)
            self.assertFalse((Path(d) / "bad").exists())

    def test_conversion_rejects_pickled_modules_with_structured_error(self):
        import contextlib
        import io
        import json
        import torch
        from recursive_mlx.cli import main
        from recursive_mlx.conversion import convert_link

        with tempfile.TemporaryDirectory() as d:
            source, destination = Path(d) / "module.pt", Path(d) / "converted"
            torch.save(torch.nn.Linear(4, 4), source)
            with self.assertRaisesRegex(ValueError, "tensor-only"):
                convert_link(source, destination, "inner", 4, 4)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = main(
                    [
                        "convert-link",
                        str(source),
                        str(destination),
                        "--kind",
                        "inner",
                        "--in-dim",
                        "4",
                        "--out-dim",
                        "4",
                    ]
                )
            self.assertEqual(status, 1)
            self.assertEqual(
                json.loads(output.getvalue())["error"]["code"], "ValueError"
            )
            self.assertFalse(destination.exists())

    def test_conversion_rejects_invalid_tensor_dictionaries(self):
        import torch
        from mlx.utils import tree_flatten
        from recursive_mlx.links import InnerLink
        from recursive_mlx.conversion import convert_link

        weights = {
            name: torch.tensor(np.array(value))
            for name, value in tree_flatten(InnerLink(4).parameters())
        }
        cases = {
            "missing": {k: v for k, v in weights.items() if k != "proj1.weight"},
            "extra": {**weights, "unexpected": torch.ones(1)},
            "nonfinite": {**weights, "proj1.weight": torch.full((4, 4), float("nan"))},
            "integer": {
                **weights,
                "proj1.weight": torch.ones((4, 4), dtype=torch.int32),
            },
            "nested": {"state_dict": weights},
        }
        with tempfile.TemporaryDirectory() as d:
            for label, state in cases.items():
                with self.subTest(case=label):
                    source, destination = Path(d) / (label + ".pt"), Path(d) / label
                    torch.save(state, source)
                    with self.assertRaises(ValueError):
                        convert_link(source, destination, "inner", 4, 4)
                    self.assertFalse(destination.exists())

    def test_smoke_real_recursive_path(self):
        from recursive_mlx.engine import smoke

        report = smoke()
        self.assertEqual(report["backend"], "mlx")
        self.assertEqual(report["rounds"], 2)
        self.assertEqual(report["qualification"], "synthetic_only")
        self.assertGreater(len(report["token_ids"]), 0)
        self.assertEqual(
            [x["role"] for x in report["stages"]],
            ["planner", "critic", "solver_feedback", "planner", "critic", "solver"],
        )


if __name__ == "__main__":
    unittest.main()
