# MLX inference preview: use cases and runbook

Read the [CTO summary](MLX_EXECUTIVE_SUMMARY.md) and [design](specs/mlx-port.md).
This is **sequential-light, math/choice prompting, batch one, uncached, greedy inference**.
Training, quantization, other styles, arbitrary remote model code, code execution and tool calls
are unsupported. Full-size checkpoints have not been qualified.

## 1. Setup and immediate proof

Use an Apple Silicon Mac and an isolated Python 3.12 environment. No Node/AXI dependency.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-mlx.txt
.venv/bin/python -m recursive_mlx
.venv/bin/python -m recursive_mlx --help
.venv/bin/python -m recursive_mlx smoke
```

The overview/help are dependency-light and do not load models. The smoke command runs
tiny **random** native MLX-LM models for planner → critic → solver feedback → planner →
critic → solver. It returns `qualification: synthetic_only`; its token IDs are not
meaningful answers. It downloads nothing. It is an execution check, not a performance
benchmark or a full-size memory estimate.

For strict reference math (use the fresh inference environment above to avoid PyTorch):
```bash
export MLX_ENABLE_TF32=0
export USE_TORCH=0
```
Set these **before** importing MLX/Transformers. The CLI supplies defaults for smoke/run.
Library callers must set them themselves before imports. `USE_TORCH=0` alone does not
prevent indirect PyTorch imports in a mixed environment; the fresh inference environment
has no PyTorch installed. Do not set USE_TORCH=0 in the independent PyTorch parity environment.

## 2. Run tests

The standard-library tests need no MLX:
```bash
python3 -m unittest discover -s tests_mlx -p test_contracts.py -v
python3 -m unittest discover -s tests_mlx -p test_recovery.py -v
```

The complete suite also needs PyTorch for independent reference math and adapter conversion.
Use a separate environment, not production inference:
```bash
uv venv --python 3.12 .venv-reference
uv pip install --python .venv-reference/bin/python -r requirements-mlx.txt 'torch==2.14.0'
MLX_ENABLE_TF32=0 USE_TORCH=1 .venv-reference/bin/python -m unittest discover -s tests_mlx -v
```

These commands install dependencies, not model checkpoints. Versions above match the
reviewed reference environment. A fresh inference environment was also installed and
verified without PyTorch; `requirements-mlx.lock.txt` records its transitive versions.
See [verification evidence and limitations](MLX_VERIFICATION.md).

## 3. Prepare a real local bundle (experimental)

This section is a supported code path tested with tiny local fixtures, **not a claim that
released full-size checkpoints were run**. Obtain operator approval and check model licenses
before acquisition. The code performs no download or automatic model conversion.

Use pre-acquired, unquantized MLX-LM-compatible safetensors for the exact
sequential-light roles: planner Qwen3, critic Llama, solver Qwen2.
Preserve their exact tokenizer/chat templates and the matching trained inner/outer links.
Do not substitute unrelated models simply because hidden dimensions match.

Convert each local PyTorch link explicitly, in the reference/conversion environment:
```bash
.venv-reference/bin/python -m recursive_mlx convert-link /approved/adapter.pt /approved/new-link \
  --kind inner --in-dim 2048 --out-dim 2048
```
**2048 is illustrative:** use the actual source/target hidden dimensions for that role.
Outer links use `--kind outer` and their actual source/target dimensions.
The converter checks key sets, shapes and finite values and refuses an existing destination.
`conversion.json` is the publication marker; weights without it are incomplete conversion.
The generated link is `weights.safetensors`.

Example directory structure (create outside the repository):
```text
bundle/
  planner/       # config.json, model safetensors and tokenizer assets
  critic/
  solver/
  planner-inner.safetensors
  critic-inner.safetensors
  solver-inner.safetensors
  outer-12.safetensors
  outer-23.safetensors
  outer-31.safetensors
  bundle.json
```

The bundle manifest uses this schema:
```json
{
  "schema": 1,
  "style": "sequential_light",
  "estimated_peak_bytes": 16000000000,
  "roles": {
    "planner": {"model": "planner", "inner": "planner-inner.safetensors", "revision": "PINNED_REVISION"},
    "critic": {"model": "critic", "inner": "critic-inner.safetensors", "revision": "PINNED_REVISION"},
    "solver": {"model": "solver", "inner": "solver-inner.safetensors", "revision": "PINNED_REVISION"}
  },
  "outer": {
    "outer_12": "outer-12.safetensors",
    "outer_23": "outer-23.safetensors",
    "outer_31": "outer-31.safetensors"
  },
  "files": {
    "planner/config.json": "SHA256_OF_FILE"
  }
}
```

The example peak is **not a measured safe budget**. Supply your approved conservative
whole-bundle estimate. Replace revision placeholders with actual immutable revisions.
The files object must contain checksums for **every model/tokenizer file and every link**;
the shortened example above deliberately will not validate.

After preparing assets, populate checksums locally:
```python
import hashlib, json
from pathlib import Path
root = Path("/approved/bundle")
path = root / "bundle.json"
manifest = json.loads(path.read_text())
manifest["files"] = {}
for p in sorted(root.rglob("*")):
    if p.is_file() and p != path:
        with p.open("rb") as stream:
            manifest["files"][p.relative_to(root).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
path.write_text(json.dumps(manifest, indent=2) + "\n")
```

Keep the bundle immutable while running. Hash verification detects prior changes; it is
not protection against a malicious concurrent writer. Store results elsewhere.

## 4. Inspect, plan, execute and resume

Input is a small JSON array, not an arbitrary Python script:
```json
[{"id":"math-001","question":"What is 17 + 25?"}]
```

```bash
.venv/bin/python -m recursive_mlx inspect /approved/bundle/bundle.json
.venv/bin/python -m recursive_mlx run \
  --bundle /approved/bundle/bundle.json --input /approved/questions.json \
  --output /approved/results/run-001 --memory-budget-bytes 20000000000 \
  --rounds 1 --latent-steps 4 --max-tokens 32 --max-context 2048 \
  --max-samples 1 --timeout-s 120 --dry-run
```

The example budget is not advice to allocate that much on your Mac. Choose approved
host-specific limits. Dry-run verifies files and request bounds without loading models
or creating output. It cannot establish actual peak memory or useful model quality.

To execute, replace `--dry-run` with `--allow-unqualified`. Run in a dedicated process.
Stdout is one structured final result; stderr reports stage progress.
Use a dedicated output directory outside the bundle, and keep the input file outside that
output directory too. New output directories are owner-only (0700); existing directories
retain their permissions, so verify those yourself before processing private inputs.
Repeat the **same command and identities** to resume previously completed samples.
Changing inputs, limits, code/dependency versions or bundle requires a new output directory.

Outputs:
- `run.json`: source/configuration/dependency/input identity.
- `sample-*.json`: immutable per-sample tokens, text and timings with integrity hash.
- `summary.json`: completed count and resume count; includes the MLX peak counter when
  this invocation executed models. A receipt-only resume reconstructs this summary without
  loading models and does not invent a peak-memory measurement.
- `.run.lock`: process exclusion, not a stale PID sentinel.

Ctrl-C/SIGTERM requests cooperative cancellation. The current sample is retried if it
did not commit a receipt. There is no mid-latent resume. Deadline includes loading and
remaining work for that invocation; already committed samples survive failures.
Non-finite decoder/link/logit values fail the sample rather than generating a successful
answer. A completed receipt survives process death; an uncommitted sample is recomputed.
SIGKILL around receipt publication and cooperative SIGTERM at a controlled loading boundary
are tested. Physical power-loss durability and interrupted Metal kernels are not certified.
Do not interpret a memory estimate or allocator setting as OS isolation.

## 5. Valuable experiments and honest interpretation

- **Port regression:** tiny PyTorch versus MLX hidden/link outputs before expensive trials.
- **Local reasoning experiments:** pinned private math/choice samples, with an independent
  evaluator. The runner records raw results; it does not implement a trusted accuracy scorer.
- **Candidate dataset labels:** only after external quality checks; generated answers are
  not ground truth.
- **Controller integration:** invoke as a bounded child process and consume receipts/progress.
  No controller, CI, Colibri or mini service is installed by this branch.

Cold-start timing includes model loading. Stage timing does not. Record both when comparing
backends. Tokens/sec alone is misleading for latent compute. Include quality, RAM, swap,
storage traffic and impact on other workloads. Never compare this tiny smoke timing to
full-checkpoint inference or claim the upstream paper's results.

## 6. Qualification status

| Capability | Evidence / limitation |
|---|---|
| Native inner/outer links | Tiny FP32 PyTorch parity with TF32 disabled |
| Qwen3/Llama/Qwen2 decoder seam | Tiny same-weight hidden-state parity |
| Recurrent latent execution | Native synthetic two-round smoke |
| Local bundle load and receipt resume | Tiny real tokenizer/safetensor integration |
| Full released checkpoints / answer quality | Not run, unqualified |
| Quantization, KV caching, batched inputs | Not implemented |
| Other styles and training | Not implemented |
| Hard OS resource isolation | Not provided |
| Process-crash recovery | SIGKILL before/after receipt rename and controlled SIGTERM tested |
| Power-loss / sustained pressure qualification | Not established; no durability certification |
| GitHub/PyPI release or model upload | Separate publication gate |
