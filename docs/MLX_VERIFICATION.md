# MLX preview verification record

Source baseline: `cbfcaab56c9a8f660a9be598750b4ff3cd762056`. Feature branch: `feat/mlx-inference-preview`.

## Executed evidence

- Apple M5, 32 GiB physical memory; bounded tiny-model tests only.
- Native smoke with randomly initialized Qwen3/Llama/Qwen2 models: two recurrent rounds, solver feedback and final greedy tokens execute on MLX.
- Tiny same-weight decoder comparisons against PyTorch pass at `atol=1e-5, rtol=1e-4` with TF32 disabled.
- Native inner/outer links pass independent PyTorch reference math at those same tolerances.
- Standard-library tests cover bounded limits, cancellation/deadline, path containment, hash tampering, malformed structured CLI errors, receipt corruption/conflict and dependency-light overview/help.
- Local integration creates tiny model safetensors and tokenizer assets, validates their checksums, performs an actual run, resumes completed samples and rejects changed run identity, duplicate sample IDs and insufficient declared budget.
- Result-directory lock exclusion is exercised across real processes. SIGKILL immediately before/after receipt rename preserves prior committed results and releases the kernel lock. Orphan staging files are ignored. This is process-crash evidence, not power-loss certification.
- A real worker receives SIGTERM at a controlled loading boundary, exits with a structured cancellation error, commits no sample, and releases its result-directory lock.
- Resume reconstructs a missing completion summary from committed receipts without loading models.
- Input/output overlap is rejected before writes; new result directories are owner-only (0700). Existing directories retain their operator-supplied permissions.
- Allocator setup failure restores memory limits and signal handlers. Non-finite decoder states, link dtype overflows and solver logits are rejected instead of producing successful answers.
- Conversion rejects wrong shapes, missing/extra keys, non-finite/integer tensors, nested dictionaries and pickled modules; refuses overwrites; and does not publish rejected conversions. Unsafe-pickle errors remain structured with no unrestricted fallback.
- A fresh isolated inference environment was installed from `requirements-mlx.txt`, without PyTorch. Native smoke and local-bundle integration passed there. Its complete installed versions are in `requirements-mlx.lock.txt`.
- The independent reference suite used Python 3.12, MLX 0.32.2, MLX-LM 0.31.3, Transformers 5.16.1, Tokenizers 0.23.2, NumPy 2.5.2, Safetensors 0.8.0 and PyTorch 2.14.0.

## Important discovered incompatibilities

1. Default MLX FP32 matmul behavior on this M5 exceeded strict link-parity tolerance. `MLX_ENABLE_TF32=0` made the unchanged test pass. This setting must precede MLX initialization; no tolerance relaxation was used.
2. MLX-LM's TokenizerWrapper is not callable; explicit `encode` is required for slot segments. The real local-bundle integration test exposed and now covers this.
3. A mixed environment with PyTorch already installed may import it indirectly through Transformers even with `USE_TORCH=0`. The no-PyTorch guarantee was verified using the separate fresh inference environment, not by relying on that environment variable.

## Reproduction

```bash
python3 -m unittest discover -s tests_mlx -p test_contracts.py -v
python3 -m unittest discover -s tests_mlx -p test_recovery.py -v
MLX_ENABLE_TF32=0 USE_TORCH=1 /path/to/reference/python -m unittest discover -s tests_mlx -v
.venv/bin/python -m recursive_mlx smoke
MLX_ENABLE_TF32=0 .venv/bin/python -m unittest discover -s tests_mlx -p test_runner.py -v
ruff check recursive_mlx tests_mlx
ruff format --check recursive_mlx tests_mlx
python3 -m compileall -q inference train recursive_mlx tests_mlx
```

## Not established

No full-size checkpoint downloads, real released-model accuracy, production latency, sustained memory behavior, physical power-loss safety, distributed recovery, hard resource isolation, end-to-end PMQTU integration, training, KV caching, quantization, other collaboration styles or external tool execution were qualified. No automatic weight upload or deployment occurred. The full design remains a staged roadmap, not a completed release claim.

The subagent review tool could not start from the parent session outside a Git repository. Review was performed directly; no independent subagent review is claimed. After operator-provided authentication became available, Jev diff review ran successfully and prioritized runner/result persistence for direct inspection. Jev also triaged the intentionally failing regression tests; its classifications are review aids, not correctness proofs. Credentials were supplied only through a child-process environment, never included in review inputs or committed.

The hardening pass expanded the reference suite from 16 to **29 passing tests**. All **21 inference/contract/recovery tests** also passed together in the separate environment with PyTorch absent and not imported; the eight reference/conversion tests require PyTorch. See [autonomous review record](MLX_AUTONOMOUS_REVIEW.md) for reproduced defects and remaining decision gates.
