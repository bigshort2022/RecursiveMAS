# CTO brief: native Apple Silicon inference preview

## Decision and delivered value

This branch adds a **native MLX sequential-light inference preview** to RecursiveMAS,
without replacing the PyTorch reference or introducing a service, broker or agent framework.

The useful deliverable is a bounded, inspectable local experiment runner: three role models
exchange latent states through native RecursiveLink modules, including recursive feedback,
then the solver generates an answer. Model weights and data stay local.
The initial interface is a Python library/CLI with JSON output and JSON progress on stderr.
Full AXI/TOON is optional and is not advertised as implemented.

## Valuable use cases

1. **Backend feasibility and regression research:** reproduce the latent computation on
   Apple Silicon and compare it to PyTorch before funding full-scale evaluation.
2. **Local model-system evaluation:** run an explicitly supplied, pinned three-role bundle
   on private math or multiple-choice questions, retain sample receipts, and resume completed work.
   Full-checkpoint quality is not yet qualified; these outputs are research candidates.
3. **Dataset experimentation:** produce candidate answers for subsequent independent
   quality checks. This does not confer trusted labels or production financial validity.
4. **Future workload-controller integration:** wrap the CLI as one bounded process with
   a manifest, memory estimate, deadline and progress stream. No PMQTU integration is
   deployed by this branch.

## Evidence and release gates

Native execution has been exercised with tiny random Qwen3, Llama and Qwen2 models.
Independent PyTorch comparisons cover tiny decoder hidden states and inner/outer links.
An offline local model-bundle test covers loading, real tokenizer prompts, inference,
durable receipts and resume. The hardened suite also exercises real process death around
receipt publication, cooperative cancellation and numerical-failure rejection.
These tests establish mechanics, **not useful answer quality**.

On the reviewed M5, default MLX FP32 matmul behavior exceeded the strict link tolerance.
The same test passed with `MLX_ENABLE_TF32=0`; tolerances were not loosened.
The reference CLI chooses that setting unless explicitly supplied otherwise, and real
runs reject a nonzero setting. Production-size and quantized behavior require separate evaluation.

Still gated: full-size released checkpoints, independent quality scoring, sustained
memory/thermal behavior, broader crash/power-loss qualification, caching/quantization,
other collaboration styles, training, sandboxed code evaluation and production deployment.
No speedup, paper reproduction or complete-spec claim is made.

## Operational economics and risks

Admission considers a declared **whole-bundle peak estimate** rather than disk capacity.
The runner has finite sample/context/round/latent/output/time bounds and cooperative
memory checks. These are not OS-level hard isolation. A running Metal kernel may delay
cancellation. Use a dedicated process under an external resource governor for real trials.

The CLI does not download models. Local artifacts must be hash-verified and unquantized.
Conversion imports local tensor-only adapter files into safetensors.
Model licensing and approved acquisitions remain operator responsibilities.
One job owns each result directory; completed sample identities cannot be overwritten
with incompatible results/configuration.

## How to run

Start with the **no-download smoke test**, not a multi-billion-parameter bundle.
See [MLX user guide](MLX_GUIDE.md) for setup, tested commands, local bundle layout and
qualification limits. The detailed design is [MLX port spec](specs/mlx-port.md).

Code remains an experimental preview until the advertised real-model gates pass.
GitHub code publication does not include weights or authorize their redistribution.
