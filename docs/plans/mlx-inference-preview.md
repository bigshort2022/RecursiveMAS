# MLX inference preview execution plan

Status: in progress. Implements `docs/specs/mlx-port.md` P0–P2 and bounded parts of P3. Real checkpoint qualification, training, caching, quantization and additional styles remain gated.

Base: cbfcaab56c9a8f660a9be598750b4ff3cd762056. Fresh clone; branch feat/mlx-inference-preview. Do not change upstream PyTorch computation.

## Ownership and order

1. Foundation: `recursive_mlx/{contracts,bundle,results}.py`; standard-library tests for bounded configuration, integrity, containment, idempotency and crash-safe receipts.
2. Native math: `recursive_mlx/{links,backend,engine}.py`; MLX tests and independent PyTorch/NumPy parity on tiny fixtures. Families: qwen3, qwen2, llama. Uncached, batch one, greedy generation only.
3. Local adapter conversion: `recursive_mlx/conversion.py`; tensor-only PyTorch import to strict MLX safetensors, new output only, no model acquisition.
4. Interface: `recursive_mlx/{__main__,cli,runner}.py`; standard-library command surface, native tiny smoke, local-bundle run, structured errors and results.
5. Docs: executive summary, explicit qualification matrix, reproducible use-case/run guide, dependency files and README pointer.

## Validation protocol

- Write tests first and record missing-feature failures.
- Use unittest (no new test dependency required).
- Run stdlib contract/CLI tests with ordinary Python; run MLX numerical tests on the available Apple Silicon environment without changing it.
- Run tiny actual decoder/link tensors, not full checkpoints or dataset downloads.
- Verify both output correctness and bounded cancellation/recovery behavior.
- Compile upstream Python to check additive import changes; preserve legacy script imports.
- Inspect diff/status and record limitations before any commit/publication.

## Acceptance

- [x] Local contract, bundle and result invariants tested.
- [x] Inner/outer links compared to independent PyTorch reference math (TF32 disabled).
- [x] Tiny required-family decoder paths execute with native MLX and same-weight PyTorch comparisons.
- [x] Sequential recurrence and final answer token generation execute offline.
- [x] Conversion validates shapes/keys/non-finite values; wrong-shape and overwrite regressions executed.
- [x] CLI help/overview do not import MLX/PyTorch or acquire models.
- [x] Local manifest path tested with actual tiny models/tokenizers; declared-budget and sample-count bounds tested. Hard OS memory enforcement is not supplied.
- [x] CTO summary/use-case guide added; overview, help, smoke, fresh inference install and test commands executed. Full-size checkpoint examples remain unqualified.
- [x] Real-model quality, training, optimization and publication gaps reported in docs/MLX_VERIFICATION.md.

## Remaining gates

- [ ] Full-size approved model bundle and independently scored quality comparison.
- [x] Targeted real-process SIGKILL receipt-boundary and cooperative SIGTERM tests.
- [ ] Broader worker crash, sustained resource-pressure and physical power-loss qualification.
- [ ] Production-size numerical tolerances and non-inferiority protocol approval.
- [ ] Remaining P4–P7 scope (optimizations, styles, training) implemented and qualified separately.
- [x] Jev-assisted risk screening and direct regression-driven hardening; see `docs/MLX_AUTONOMOUS_REVIEW.md`.
- [ ] Independent code review before merge or release.

The implementation is an experimental inference preview, not completion of the whole design.
