import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ContractTests(unittest.TestCase):
    def test_limits_reject_unbounded_values(self):
        from recursive_mlx.contracts import Limits

        for kwargs in (
            {"rounds": 0},
            {"latent_steps": -1},
            {"max_tokens": 0},
            {"timeout_s": float("inf")},
            {"rounds": True},
            {"max_context": 0},
            {"max_samples": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Limits(**kwargs)
        self.assertEqual(Limits().rounds, 1)

    def test_cancellation_and_deadline(self):
        from recursive_mlx.contracts import Budget, Cancelled, Limits

        clock = [0.0]
        b = Budget(Limits(timeout_s=1), clock=lambda: clock[0])
        clock[0] = 2
        with self.assertRaises(TimeoutError):
            b.check()
        b = Budget(Limits(), cancelled=lambda: True)
        with self.assertRaises(Cancelled):
            b.check()

    def test_receipt_identity_and_corruption(self):
        from recursive_mlx.results import Receipts

        with tempfile.TemporaryDirectory() as d:
            r = Receipts(Path(d), {"bundle": "abc", "seed": 1})
            self.assertIsNone(r.get("sample"))
            r.put("sample", {"text": "answer"})
            self.assertEqual(r.get("sample"), {"text": "answer"})
            r.put("sample", {"text": "answer"})
            with self.assertRaises(ValueError):
                r.put("sample", {"text": "changed"})
            with self.assertRaises(ValueError):
                Receipts(Path(d), {"bundle": "other"})
            path = next(Path(d).glob("sample-*.json"))
            path.write_text("{}")
            with self.assertRaises(ValueError):
                r.get("sample")

    def test_bundle_containment_and_hash(self):
        from recursive_mlx.bundle import checked_file

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = root / "x"
            p.write_bytes(b"hello")
            digest = hashlib.sha256(b"hello").hexdigest()
            self.assertEqual(checked_file(root, "x", digest), p.resolve())
            with self.assertRaises(ValueError):
                checked_file(root, "../x", digest)
            with self.assertRaises(ValueError):
                checked_file(root, "x", "0" * 64)

    def test_cli_no_heavy_imports(self):
        p = subprocess.run(
            [
                sys.executable,
                "-c",
                "from recursive_mlx.cli import main; import sys; "
                "main([]); assert 'mlx.core' not in sys.modules; assert 'torch' not in sys.modules",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads(p.stdout)["qualification"], "experimental")

    def test_bundle_malformed_is_structured(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bundle.json"
            for content in (
                "[]",
                '{"schema":1,"style":"sequential_light","roles":null}',
            ):
                path.write_text(content)
                p = subprocess.run(
                    [sys.executable, "-m", "recursive_mlx", "inspect", str(path)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(p.returncode, 1)
                self.assertIn("error", json.loads(p.stdout))
                self.assertNotIn("Traceback", p.stderr)

    def test_cli_usage_and_help(self):
        for args, expected in [(["run"], 2), (["--help"], 0), (["run", "--help"], 0)]:
            p = subprocess.run(
                [sys.executable, "-m", "recursive_mlx", *args],
                capture_output=True,
                text=True,
            )
            self.assertEqual(p.returncode, expected, p.stderr)
            if expected:
                self.assertEqual(json.loads(p.stdout)["error"]["code"], "usage")


if __name__ == "__main__":
    unittest.main()
