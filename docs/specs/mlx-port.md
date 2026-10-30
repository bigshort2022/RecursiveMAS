# RecursiveMAS native MLX port and GitHub publication

Status: proposed design for review. Specification only; not a claim of implemented MLX support, permission to publish model weights, or approval to download/train models.

Target source repository: `bigshort2022/RecursiveMAS`. Initial reviewed reference: `cbfcaab56c9a8f660a9be598750b4ff3cd762056`; reconcile and pin the actual implementation base before editing. This local planning document must move to an authoritative reviewed branch in that repository before implementation. The PMQTU workload controller consumes the eventual backend as an optional adapter; it is not a runtime dependency of this project.


## CTO executive summary

**Opportunity:** enable bounded latent multi-agent inference on Apple Silicon without a CUDA service, while preserving the PyTorch research reference. The business value is a reproducible local experimentation platform—not a promised speedup or a substitute for validated domain models.

**Delivery:** an additive sequential-light inference preview, local artifact verification/conversion, native MLX links and decoder execution, bounded runs and durable sample results. Full-scale checkpoint quality and memory qualification remain explicit release gates. Training and other collaboration styles are later projects.

**Valuable use cases:** compare single-model versus recursive reasoning on frozen math/science tasks; evaluate candidate research models privately; produce candidate synthetic answers for independently validated datasets; and integrate bounded evaluation as a workload-controller job. No generated-code execution, financial decision automation or unsupervised publication of synthetic labels is authorized.

**Architecture decision:** retain upstream; use Python plus structured JSON/JSONL, with AXI optional. Avoid a new queue, service or framework dependency. A dedicated process owns model memory and cancellation.

**Go/no-go criteria:** numerical parity, task-quality evidence, measured host-memory/cancellation behavior and honest capability reporting. Do not deploy larger bundles, quantization or caching solely because a tiny smoke test passes. Report unsupported and unqualified separately.

**Publication:** feature branch and reviewable PR in the fork; no weights or private datasets in Git. The user guide must distinguish tested synthetic execution from externally supplied model-bundle runs.

## 1. Decision and alternatives

**Recommendation:** an additive MLX inference backend in the existing fork, retaining the PyTorch implementation as the reference. Ship a normal Python library and CLI with structured JSON/JSONL output. Full AXI is optional follow-up packaging, not a required MLX dependency or a reason to delay numerical qualification.

Alternatives:

- Separate MLX-only repository: simpler dependencies and branding, but duplicate lifecycle, docs and fixtures, and increased upstream drift. Defer unless upstream compatibility becomes unmanageable.
- Rewrite the entire project in MLX: consistent native implementation, but risks breaking the research reference and combining inference, training and evaluation migrations. Reject for the first release.
- Add a full AXI wrapper first: convenient for agents, but does not establish native latent execution or quality. Preserve AXI-friendly semantics without making TOON or an agent framework mandatory.

An additive implementation is the best initial fit for a research repository serving both human researchers and agents. This is a design judgment, not a measured preference survey.

## 2. Goals, scope and honest naming

The first release must run a complete **sequential-light RecursiveMAS inference path** using native MLX arrays, modules and model computation on Apple Silicon. It must include trained inner/outer links and recurrent latent exchange, not substitute text messages or ordinary chat completion calls.

Initial target bundle: the existing sequential-light planner/critic/solver configurations. Their exact Qwen3, Llama and Qwen2-family model revisions, tokenizer assets and inner/outer links must be pinned in a bundle manifest. A different collection cannot silently substitute for a missing or unsupported role.

Initial scope:

- Native inner/outer RecursiveLink modules.
- Supported-family embeddings, final normalized decoder hidden states and output logits.
- Correct sequential latent stages, recurrence and final answer generation.
- Conversion/import of explicitly approved local artifacts.
- Tokenization/prompt parity and evaluation parity.
- Bounded memory, cancellation, sample-level results, diagnostics and qualification reports.

Later milestones, each separately qualified: sequential-scaled; mixture; distillation; deliberation and controlled tool calls; inner-link training; outer-link training. The first publication must say **MLX inference preview: sequential-light**, not 'full RecursiveMAS port'.

Non-goals for the first release: distributed inference, arbitrary model families, streaming huge Colibri-format weights, production service deployment, model downloads on import, training, unsafe code evaluation, automatic public checkpoint publication, and guaranteed speedup over PyTorch.

## 3. Evidence and feasibility boundary

The reviewed upstream implementation requires direct `inputs_embeds` and final decoder hidden states, trained adapters and role-specific tokenization. Existing chat APIs are insufficient.

During review, the upstream `mlx-lm` Qwen3 model source exposed `input_embeddings`, its normalized decoder result and per-layer caches. This is evidence of one useful seam, NOT proof all required model families or pinned releases support identical APIs. Before implementation, pin MLX/MLX-LM versions and inspect/test every required family. Do not depend on mutable GitHub HEAD behavior.

The existing RecursiveMAS license was inspected as MIT with copyright attribution. Model weights, tokenizer files, datasets and gated assets have separate licensing/access conditions. Code licensing does not authorize redistribution of those assets.

## 4. Architecture and isolation

Maintain the PyTorch commands and results as the reference surface. Add a backend-specific package with a narrow public API and a CLI. Exact package placement is resolved after repository instruction/layout review, without a broad unrelated reorganization.

Modules have distinct responsibilities:

- **Bundle manifest and capability registry:** identities, file hashes, role graph, model families, supported operation matrix and qualification state.
- **Conversion utilities:** weights/link import, explicit schema mapping and reports; conversion-only dependencies isolated from runtime.
- **MLX family adapters:** embeddings, normalized decoder execution, logits and cache operations.
- **RecursiveLink modules:** native MLX implementations of the trained inner/outer math.
- **Execution engine:** stage sequencing, budgets, caches and final generation.
- **Results and evaluation:** per-sample receipts, scoring, aggregate reports and replay/resume.
- **CLI/API:** one implementation of validation and execution, with human and machine output adapters.

The MLX inference runtime must not import PyTorch merely to start, inspect capabilities, load converted safetensors or run inference. Conversion/reference-test environments may contain PyTorch. Optional extras or separate lockable environments keep those dependencies explicit. Heavy imports are lazy; CLI help and inspect do not initialize Metal or load models.

Runtime never requires PMQTU, NATS, Docker, an AXI package or a cloud service. PMQTU integration wraps it externally.

## 5. Backend semantic contract

Each supported family must implement and test:

- Token IDs → embeddings with exact tokenizer/chat-template identity.
- Embeddings → final post-normalization decoder states with correct causal masking, positions and padding behavior.
- Final states → logits, respecting tied embeddings and family-specific output behavior.
- Cache allocation, update, reset and release with explicit offset semantics.
- Declared dtype/quantization support and structured errors for unsupported operations.

Initial execution is batch size one to reduce padding/cache complexity and loading peaks. Batch support follows only after left/right padding, unequal lengths, masked positions and per-sample stopping tests pass. Do not simulate unsupported masks by silently dropping padding or altering prompts.

Inner links preserve LayerNorm epsilon, exact versus approximate GELU choice, biases, residual order and weight layout. Outer links preserve source/target normalization, projection dimensions, residual projection and module dtype. Inspect source/config values rather than copying framework defaults. Converted shape compatibility is necessary but not evidence of semantic compatibility.

Preserve the number/order of latent steps and rounds, role graph, prompt injection points, final feedback semantics and answer extraction. Do not infer semantics from class names or replace an architecture because its hidden width matches.

### Baseline before optimization

First implement an uncached MLX reference path matching the current uncached PyTorch latent computation. Then add cached execution as a separately gated optimization. Cache reuse is allowed only within the same unchanged causal prefix, role/model, mask, position sequence and dtype profile. Round/role transitions reset caches unless a validated contract proves reuse safe.

Validate the initial prefill and every subsequent latent append. Compare both hidden states and final outputs. Sliding-window/hybrid/recurrent caches require family-specific qualification; a generic key/value cache is not presumed sufficient.

## 6. Artifacts, conversion and reproducibility

A bundle manifest records:

- schema version and immutable bundle ID;
- source repository commit and conversion tool version;
- each model/tokenizer revision, file hashes, architecture and licensing/access notes;
- inner and outer adapter hashes, configuration and role-edge mapping;
- source/target tensor names, shapes, layouts and dtypes;
- quantization method/group size and excluded tensors;
- prompt/chat-template versions and special-token IDs;
- supported backend versions and qualification report references.

Conversion is a separate explicit operation, local-files-only by default, with source/output size estimates and bounded memory. It writes a new staging destination, validates mappings and commits a manifest only after all required outputs verify. Never overwrite the input bundle, mutate a shared model cache or expose a partially converted bundle as ready.

Prefer safetensors for runtime weights and strict JSON metadata. Import `.pt` adapters only through a reviewed conversion path using safe tensor-only loading where supported; never fall back silently to unrestricted pickle execution. Reject unknown/missing keys, unsupported layouts, mismatched edges and non-finite weights. A deterministic conversion report lists every intentional transformation.

Separate profiles:

1. Unquantized FP32 small fixtures for semantic parity.
2. Unquantized FP16/BF16 hardware-supported production-size parity, with measured tolerances.
3. Quantized base models, qualified separately for task quality. Keep links unquantized initially unless a separate experiment validates their quantization.

No automatic 'latest' resolution during a run. Pin model and tokenizer revisions together. Changing a link, model, quantizer, prompt or runtime produces a new execution identity even if file names remain unchanged.

## 7. Memory, execution and cancellation

Use one owned worker process per active MAS session initially, with at most one active request and the entire role bundle included in admission. Account for converted weights, link modules, prompts, cache, activations, lazy evaluation and conversion/loading transients. Disk size is not the RAM estimate.

Every run requires finite bounds on samples, rollouts, recursive rounds, latent steps, context length, generated tokens and wall time. Validate combinations before loading. Reject unknown required capacity or provide an explicit dry-run estimate with uncertainty; do not silently proceed because the model fits on disk.

MLX execution is lazy: materialize/evaluate results at documented stage boundaries to bound retained graphs and ensure timing measures completed computation. Use memory/peak/cache counters exposed by the pinned MLX release with source-labeled semantics; report host RSS/pressure/swap separately. Never add overlapping counters as though they were distinct memory pools.

Cancellation is cooperative between bounded stages or generated-token steps, with process-level termination available after an explicit grace period. A running Metal operation may not be immediately interruptible. The report must distinguish cancel requested, cancelling and confirmed stopped.

Idle teardown releases owning references and exits the dedicated process when necessary. Verify process exit and observed resource behavior, not merely cache-clear return. Do not promise immediate disappearance of filesystem cache or host compressed memory. No default always-on model service in this repository.

## 8. Outputs, resume and telemetry

Emit durable per-sample results identified by bundle, dataset/sample ID, prompt/configuration hash, seed, rollout and evaluator version. Publish completed sample receipts atomically and derive aggregate scores from validated receipts. A crash before receipt publication leaves the sample incomplete and eligible for a bounded restart; no mid-latent checkpoint/resume claim in v1.

Resume refuses mismatched identities, corrupted receipts and incompatible evaluation settings. User-request idempotency and stochastic replicate identity are separate concepts; a new replicate must not accidentally reuse a cached sample.

Machine events use a versioned JSONL envelope with run/sample IDs, role, phase, round, completed/total units where known, elapsed time, resource measurements and error/cancellation codes. Human progress goes to stderr; stdout remains clean structured output when machine mode is selected.

Measure load, tokenization, latent stages, final generation and evaluation separately. Include cold/warm status, cache profile, input/latent/output counts, dtype/quantization, peak memory and host interference. Do not call latent iterations output tokens or compare unlike throughput metrics.

Raw prompts, gold labels, generated outputs and latent arrays are not routine telemetry. Persist them only as authorized result/debug artifacts. Latent trajectories are not human-readable reasoning; never present a generated explanation as an exact account of latent computation.

## 9. API, CLI and AXI decision

Core public operations: inspect bundle/capabilities, validate, estimate, run bounded evaluation, inspect results, resume completed-sample work and benchmark. Names and import paths are pinned in the implementation plan after packaging review; this specification does not claim those APIs already exist.

CLI behavior:

- No arguments: compact local capability/status summary without network, loading or acquisition.
- Scoped help with examples, defaults and explicit supported styles/backends.
- Noninteractive operation; stable exit codes (0 success/no-op, 1 execution failure, 2 usage error).
- Structured JSON errors and JSON/JSONL output options; bounded summaries and explicit truncation.
- Separate local conversion/acquisition from execution. Network-disabled execution is supported.
- `--dry-run` reports required artifacts, capacities and blockers without starting computation.

These are AXI-friendly behaviors, but this release is not labeled fully AXI-compliant without TOON and conformance checks. An optional AXI facade may later render the same result schema as TOON and supply generated skills/context hooks. It must add no numerical logic, runtime dependency or divergent policy. Do not make NumPy/MLX users install Node merely to run inference.

## 10. Numerical and task-quality gates

Do not require bitwise equality between frameworks or identical random sampling sequences. Do not relax tolerances after seeing failures without a reviewed, documented explanation.

Gate A: module fixtures. On finite FP32 tiny fixtures, initial acceptance is elementwise `abs(error) <= 1e-5 + 1e-4 * abs(reference)` for isolated link modules. Test GELU, normalization, residuals, biases, non-square outer projections, shape errors and extreme but finite inputs. Pin these thresholds before execution; any change requires review.

Gate B: tiny full decoder fixtures. Compare embeddings, post-norm states, latent trajectories and logits for every supported family across empty/short/long boundaries, special tokens, positions and supported masks. Define family/dtype tolerances from a frozen numerical qualification proposal before running the scored comparison; production model-family support stays unqualified until that proposal and evidence exist. Tiny synthetic fixtures must run independently of gated model downloads.

Gate C: end-to-end sequential-light. With pinned prompts and deterministic generation, compare stage identities, tokenization and evaluated answers. Explain near-tie token differences rather than calling them automatic semantic failures or ignoring them. Run multiple representative prompts and pathological input tests, not one demo.

Gate D: quality non-inferiority. Freeze a disjoint evaluation set, metric, number of repeated runs and an operator-approved degradation margin before comparing quantized/cached profiles. Report paired differences and uncertainty; a small canary cannot qualify broad quality claims. Avoid fitting thresholds on the test set. Separate changes from backend port, quantization and cache optimization.

Gate E: resource qualification. Bound memory, cancellation latency and storage interference on the intended Mac. Record model loading transients, pressure/swap and interference with protected workloads. Performance gates depend on measured host budgets, not invented free RAM.

Gate F: regression. Existing PyTorch interfaces/tests remain functional; offline contract and conversion tests pass without model acquisition. Actual Metal execution must be tested on Apple Silicon, not inferred from Linux CI success.

## 11. Benchmarks and publication claims

Compare at least:

- current PyTorch reference with explicit device and precision;
- uncached MLX semantic baseline;
- cached MLX after parity qualification;
- quantized MLX after separate quality qualification;
- a single-model baseline for whether MAS is beneficial at all.

Use the same input set, evaluation policy and declared resource budget. Record wall time, quality, per-stage latency, peak memory, disk reads and cold/warm behavior. Synchronize lazy/asynchronous execution at timing boundaries. Include loading, evaluation and network time explicitly rather than excluding them silently.

No promised 10x speedup, automatic claim of paper reproduction or claim that MLX universally outperforms CUDA. Publish results only with hardware, macOS, source/lock revisions, configuration, sample count and uncertainty. Failed or unsupported configurations are reportable outcomes.

## 12. Security and tool-enabled styles

Default to no remote model code, no network acquisition and no generated-code evaluation. Pinned reviewed model code may be enabled explicitly with provenance. Keep secrets out of manifests, CLI output and repository artifacts.

Deliberation/tool-calling is a later capability. Tool requests must pass an allowlisted broker/adapter with spend/rate limits, timeouts and auditable results. Do not directly adopt plaintext key-file examples as production configuration.

Code benchmarks require independently qualified isolation for filesystem, network, process count, CPU/RAM and credentials. A timeout or subprocess is not that isolation. Never mount the 19 TB production data writable into generated-code evaluation.

## 13. Staged delivery and team boundaries

P0 — reference capture and compatibility inventory: pin source/MLX versions, inspect required families and current tests, define bundle schema, safe local fixtures and licensing report. Record supported, unsupported and unqualified separately.

P1 — foundation parity: native links, conversion schema, tiny family adapter interfaces and fixture tests. One contract owner; do not parallelize consumers before stable contracts.

P2 — first inference vertical slice: sequential-light uncached, batch one, local pinned artifacts, bounded run, per-sample receipt and process teardown. No full training or AXI facade.

P3 — qualification: production-size unquantized parity, task scoring, Apple Silicon memory/cancellation tests and PyTorch regression. Publish no qualified model-family claim until passed.

P4 — optimizations: independent proposals for cache reuse, batching and quantization, each compared against P3 and reviewed before becoming default. No combined optimization that obscures the source of quality regression.

P5 — GitHub inference preview: documentation, capability matrix, tested examples, clean dependency separation and reproducible benchmark reports. Optional AXI facade is a separate nonblocking change only if agent users need it.

P6 — further collaboration styles: add one at a time with matching bundles, fixtures and quality gates; do not advertise full coverage prematurely.

P7 — training port as a separate approved project: native gradient parity and optimization semantics for inner links first, then outer-link training with frozen bases, optimizer state/checkpoint/resume, RNG/dtype consistency and loss/gradient comparisons on tiny fixtures. Training memory and optimizer peaks need separate admission estimates. Inference checkpoint compatibility does not prove training equivalence.

Parallel implementation begins only after file ownership and dependency plans are committed. Natural later parallel scopes: family adapters, result/CLI views, conversion validation and benchmark harness, provided shared interfaces and files are frozen. Integration owner alone changes package metadata, shared exports and CLI registration.

## 14. GitHub upload/release protocol

Use `bigshort2022/RecursiveMAS` as the proposed destination, on a dedicated feature branch and reviewable PR. Confirm remote branch state and inspect repository instructions before edits. Retain upstream attribution and PyTorch reference; do not force-push or replace the default branch.

Publication stages:

1. Commit reviewed spec and executable plans in the fork.
2. Open a draft implementation PR clearly labeled experimental/incomplete while gates remain open.
3. Run offline CI for schemas, converters and API behavior. Use a trusted Apple Silicon runner or independently captured hardware evidence for Metal qualification; never expose a privileged self-hosted runner to untrusted PR code.
4. Review dependency locks, licenses, secret scans, generated artifacts and the exact staged diff.
5. Publish the code/docs PR and evidence without weights, private datasets, credentials, user paths or runtime logs containing payloads.
6. Tag a preview only when the advertised capability gates pass and the owner approves release. A failed hardware gate keeps the PR experimental.

Code upload and model-weight upload are separate actions. Do not commit multi-GB weights to GitHub. Converted weights may be published to an explicitly approved model-hosting repository only after checking each source license/gating requirement, destination authorization, conversion provenance, model card and checksum manifest. Some weights may require users to convert locally instead of redistribution.

The README must state exact supported styles/models, inference-versus-training scope, tested Macs, memory observations, unsupported features and how to reproduce results. Avoid changing branding to imply official upstream endorsement.

## 15. Integration with the workload-control specification

Expose the MLX port through the proposed `recursive_mas.evaluate` adapter after local qualification. The outer controller owns cross-job admission, process lifecycle and campaign budget. The MLX worker owns bounded intra-job stages and model caches. One authority owns idle teardown; no nested competing supervisors.

The adapter records bundle/configuration identity, consumes immutable dataset references and returns result receipts and progress events. It reports restart-only versus sample-resumable explicitly. JetStream never carries per-step latent tensors. Colibri remains a separate runtime unless it implements the required latent contract and compatible trained links; ordinary chat compatibility is insufficient.

## 16. Open release gates and review checklist

These are explicit evidence gates, not invented implementation assumptions:

- Exact implementation branch and dependency pins.
- Required-family MLX capability and prompt/tokenizer compatibility.
- Approved source model revisions and acquisition size/budget.
- Frozen full-decoder numerical tolerances and quality evaluation protocol.
- Hardware resource limits, benchmark budget and permitted host.
- Confirmed destination/visibility and authorization for any converted-weight upload.

Self-review requirements: every advertised capability maps to a test/evidence gate; unsupported training/styles are visible; no conversion changes source assets; cache optimization is separate from semantic baseline; structured output does not require AXI; GitHub code publication cannot imply model redistribution authorization.
