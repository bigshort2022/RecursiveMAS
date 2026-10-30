import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
from mlx.utils import tree_flatten


def make_bundle(root):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    from recursive_mlx.backend import tiny_backend
    from recursive_mlx.links import InnerLink, OuterLink
    from recursive_mlx.bundle import sha256

    roles = {}
    for role, family in (
        ("planner", "qwen3"),
        ("critic", "llama"),
        ("solver", "qwen2"),
    ):
        directory = root / role
        directory.mkdir()
        b = tiny_backend(family, 8)
        mx.save_safetensors(
            str(directory / "model.safetensors"),
            dict(tree_flatten(b.model.parameters())),
        )
        (directory / "config.json").write_text(
            json.dumps(dataclasses.asdict(b.model.args))
        )
        t = Tokenizer(
            WordLevel({"[UNK]": 0, "[EOS]": 1, "hello": 2}, unk_token="[UNK]")
        )
        t.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=t, unk_token="[UNK]", eos_token="[EOS]"
        )
        tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
        tokenizer.save_pretrained(directory)
        link = root / (role + ".safetensors")
        mx.save_safetensors(str(link), dict(tree_flatten(InnerLink(8).parameters())))
        roles[role] = {
            "model": role,
            "inner": link.name,
            "revision": "synthetic-fixture",
        }
    outer = {}
    for edge in ("outer_12", "outer_23", "outer_31"):
        p = root / (edge + ".safetensors")
        mx.save_safetensors(str(p), dict(tree_flatten(OuterLink(8, 8).parameters())))
        outer[edge] = p.name
    data = {
        "schema": 1,
        "style": "sequential_light",
        "roles": roles,
        "outer": outer,
        "estimated_peak_bytes": 1000000,
        "files": {
            p.relative_to(root).as_posix(): sha256(p)
            for p in root.rglob("*")
            if p.is_file()
        },
    }
    manifest = root / "bundle.json"
    manifest.write_text(json.dumps(data))
    return manifest


class RunnerTests(unittest.TestCase):
    def test_local_bundle_dry_run_run_and_resume(self):
        from recursive_mlx.runner import run
        from recursive_mlx.contracts import Limits

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle_dir = root / "bundle"
            bundle_dir.mkdir()
            manifest = make_bundle(bundle_dir)
            samples = root / "input.json"
            samples.write_text(json.dumps([{"id": "one", "question": "What is 1+1?"}]))
            args = SimpleNamespace(
                bundle=manifest,
                input=samples,
                output=root / "output",
                memory_budget_bytes=50000000,
                dry_run=True,
                allow_unqualified=True,
            )
            limits = Limits(max_tokens=2, max_context=256, latent_steps=1)
            report = run(args, limits)
            self.assertEqual(report["status"], "planned")
            self.assertFalse(args.output.exists())
            args.dry_run = False
            report = run(args, limits)
            self.assertEqual(report["completed"], 1)
            self.assertEqual(report["resumed"], 0)
            # Simulate death after the last receipt but before summary publication.
            (args.output / "summary.json").unlink()
            from unittest.mock import patch

            with patch(
                "recursive_mlx.backend.load_backend",
                side_effect=AssertionError("resume must not load models"),
            ):
                report = run(args, limits)
            self.assertEqual(
                json.loads((args.output / "summary.json").read_text()), report
            )
            self.assertEqual(report["resumed"], 1)
            self.assertEqual(report["model_quality"], "unqualified")
            with self.assertRaises(ValueError):
                run(args, Limits(max_tokens=3, max_context=256, latent_steps=1))

    def test_preflight_rejects_budget_and_duplicate_ids(self):
        from recursive_mlx.runner import run
        from recursive_mlx.contracts import Limits

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "bundle"
            directory.mkdir()
            manifest = make_bundle(directory)
            samples = root / "input.json"
            samples.write_text(
                '[{"id":"a","question":"hello"},{"id":"a","question":"hello"}]'
            )
            args = SimpleNamespace(
                bundle=manifest,
                input=samples,
                output=root / "out",
                memory_budget_bytes=100,
                dry_run=True,
                allow_unqualified=True,
            )
            with self.assertRaises(ValueError):
                run(args, Limits(max_samples=2))
            args.memory_budget_bytes = 50000000
            with self.assertRaises(ValueError):
                run(args, Limits(max_samples=2))
            self.assertFalse(args.output.exists())

    def test_preflight_rejects_input_within_output_directory(self):
        from recursive_mlx.runner import run
        from recursive_mlx.contracts import Limits

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "bundle"
            directory.mkdir()
            manifest = make_bundle(directory)
            output = root / "output"
            output.mkdir()
            samples = output / "summary.json"
            original = '[{"id":"a","question":"hello"}]'
            samples.write_text(original)
            args = SimpleNamespace(
                bundle=manifest,
                input=samples,
                output=output,
                memory_budget_bytes=50000000,
                dry_run=True,
                allow_unqualified=True,
            )
            with self.assertRaisesRegex(ValueError, "input.*outside"):
                run(args, Limits())
            self.assertEqual(samples.read_text(), original)
            self.assertEqual(list(output.iterdir()), [samples])

    def test_allocator_setup_failure_restores_process_state(self):
        import signal
        from unittest.mock import patch
        from recursive_mlx.runner import run
        from recursive_mlx.contracts import Limits

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "bundle"
            directory.mkdir()
            manifest = make_bundle(directory)
            samples = root / "input.json"
            samples.write_text('[{"id":"a","question":"hello"}]')
            args = SimpleNamespace(
                bundle=manifest,
                input=samples,
                output=root / "out",
                memory_budget_bytes=50000000,
                dry_run=False,
                allow_unqualified=True,
            )
            handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
            old_limit = mx.set_memory_limit(60000000)
            mx.set_memory_limit(old_limit)
            try:
                with patch(
                    "mlx.core.set_cache_limit",
                    side_effect=RuntimeError("injected allocator failure"),
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "injected allocator failure"
                    ):
                        run(args, Limits())
                observed_limit = mx.set_memory_limit(old_limit)
                self.assertEqual(observed_limit, old_limit)
                for sig, handler in handlers.items():
                    self.assertEqual(signal.getsignal(sig), handler)
            finally:
                mx.set_memory_limit(old_limit)
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)

    def test_sigterm_cancels_worker_without_committing_sample(self):
        import select
        import signal
        import subprocess
        import sys
        from recursive_mlx.results import RunLock

        worker_code = """
import signal
import threading
from recursive_mlx import backend
from recursive_mlx.cli import main
load = backend.load_backend
def controlled_load(path):
    handler = signal.getsignal(signal.SIGTERM)
    received = threading.Event()
    def forward(*args):
        handler(*args)
        received.set()
    signal.signal(signal.SIGTERM, forward)
    try:
        print("ready", flush=True)
        if not received.wait(10):
            raise RuntimeError("test cancellation did not arrive")
    finally:
        signal.signal(signal.SIGTERM, handler)
    return load(path)
backend.load_backend = controlled_load
raise SystemExit(main())
"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "bundle"
            directory.mkdir()
            manifest = make_bundle(directory)
            samples, output = root / "input.json", root / "out"
            samples.write_text('[{"id":"a","question":"hello"}]')
            worker = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    worker_code,
                    "run",
                    "--bundle",
                    str(manifest),
                    "--input",
                    str(samples),
                    "--output",
                    str(output),
                    "--memory-budget-bytes",
                    "50000000",
                    "--allow-unqualified",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                ready, _, _ = select.select([worker.stdout], [], [], 15)
                self.assertTrue(ready, "worker did not reach cancellation boundary")
                self.assertEqual(worker.stdout.readline().strip(), "ready")
                worker.send_signal(signal.SIGTERM)
                stdout, stderr = worker.communicate(timeout=15)
                self.assertEqual(worker.returncode, 1, stderr)
                self.assertEqual(json.loads(stdout)["error"]["code"], "Cancelled")
                self.assertTrue((output / "run.json").exists())
                self.assertEqual(list(output.glob("sample-*.json")), [])
                self.assertFalse((output / "summary.json").exists())
                with RunLock(output):
                    pass
            finally:
                if worker.poll() is None:
                    worker.kill()
                worker.communicate(timeout=10)

    def test_resume_lock_exclusion(self):
        from recursive_mlx.results import RunLock

        with tempfile.TemporaryDirectory() as d:
            with RunLock(d):
                with self.assertRaises(RuntimeError):
                    with RunLock(d):
                        pass


if __name__ == "__main__":
    unittest.main()
