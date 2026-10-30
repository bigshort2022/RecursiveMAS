"""Single-directory, immutable per-sample receipts with process exclusion."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(canonical(data) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class Receipts:
    """Caller must hold RunLock when mutating; each committed sample is immutable."""

    def __init__(self, root, run_identity):
        self.root = Path(root)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.run_hash = identity(run_identity)
        path = self.root / "run.json"
        record = {"schema": 1, "identity": run_identity, "run_hash": self.run_hash}
        if path.exists():
            if json.loads(path.read_text()) != record:
                raise ValueError("resume identity mismatch")
        else:
            atomic_json(path, record)

    def _path(self, sample_id):
        return self.root / ("sample-" + identity(sample_id) + ".json")

    def get(self, sample_id):
        path = self._path(sample_id)
        if not path.exists():
            return None
        row = json.loads(path.read_text())
        if (
            not isinstance(row, dict)
            or row.get("run_hash") != self.run_hash
            or row.get("sample_id") != sample_id
            or not isinstance(row.get("result"), dict)
            or row.get("result_hash") != identity(row["result"])
        ):
            raise ValueError("corrupt or mismatched sample receipt")
        return row["result"]

    def put(self, sample_id, result):
        if not isinstance(result, dict):
            raise ValueError("sample result must be an object")
        old = self.get(sample_id)
        if old is not None:
            if old != result:
                raise ValueError("conflicting sample result")
            return
        atomic_json(
            self._path(sample_id),
            {
                "run_hash": self.run_hash,
                "sample_id": sample_id,
                "result": result,
                "result_hash": identity(result),
            },
        )


class RunLock:
    def __init__(self, root):
        self.root = Path(root)

    def __enter__(self):
        import fcntl

        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.stream = (self.root / ".run.lock").open("a+")
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise RuntimeError("another worker owns this result directory") from None
        return self

    def __exit__(self, *exc):
        self.stream.close()
