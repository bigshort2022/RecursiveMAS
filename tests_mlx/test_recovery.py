"""Real subprocess death at deterministic receipt publication boundaries.

These are process-crash tests, not power-loss or filesystem durability certification.
"""

import select
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from recursive_mlx.results import Receipts, RunLock


WORKER = r"""
import signal
import sys
from recursive_mlx import results

root, phase = sys.argv[1:]
with results.RunLock(root):
    receipts = results.Receipts(root, {"bundle": "crash-fixture"})
    receipts.put("stable", {"text": "committed"})
    replace = results.os.replace
    def crash_boundary(source, target):
        if phase == "after_replace":
            replace(source, target)
        print("ready", flush=True)
        signal.pause()
        raise AssertionError("worker must be killed at the publication boundary")
    results.os.replace = crash_boundary
    receipts.put("interrupted", {"text": "new"})
"""


@unittest.skipUnless(
    sys.platform == "darwin", "preview targets macOS flock/filesystem semantics"
)
class RecoveryTests(unittest.TestCase):
    def test_sigkill_before_receipt_replace_preserves_committed_samples(self):
        self.assert_crash_recovery("before_replace", None)

    def test_sigkill_after_receipt_replace_retains_complete_receipt(self):
        self.assert_crash_recovery("after_replace", {"text": "new"})

    def assert_crash_recovery(self, phase, expected):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            worker = subprocess.Popen(
                [sys.executable, "-c", WORKER, str(root), phase],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                ready, _, _ = select.select([worker.stdout], [], [], 10)
                self.assertTrue(ready, "worker failed to reach publication boundary")
                self.assertEqual(worker.stdout.readline().strip(), "ready")
                with self.assertRaises(RuntimeError):
                    with RunLock(root):
                        pass
                worker.kill()
                _, stderr = worker.communicate(timeout=10)
                self.assertEqual(worker.returncode, -signal.SIGKILL, stderr)
                # Kernel exclusion is released by process death, not a guessed timeout.
                with RunLock(root):
                    receipts = Receipts(root, {"bundle": "crash-fixture"})
                    self.assertEqual(receipts.get("stable"), {"text": "committed"})
                    self.assertEqual(receipts.get("interrupted"), expected)
                    # Orphan .pending-* files are not mistaken for completed samples.
                    receipts.put("interrupted", {"text": "new"})
                    self.assertEqual(receipts.get("interrupted"), {"text": "new"})
            finally:
                if worker.poll() is None:
                    worker.kill()
                worker.communicate(timeout=10)


if __name__ == "__main__":
    unittest.main()
