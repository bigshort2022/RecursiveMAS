# Autonomous review and hardening handoff

Scope: draft MLX inference preview following initial commit `b8ed0af`.
The operator authorized autonomous follow-up, not deployment or removal of qualification gates.

## Executive outcome

The bounded preview now has stronger failure handling and recovery evidence. The reference
suite expanded from 16 to 29 passing tests; all 21 non-reference tests also passed together
in the separate PyTorch-free environment. No full-size checkpoint was acquired; useful
answer quality, production performance and the full staged specification remain unqualified.

Jev authentication was supplied privately through a child-process environment. No key was
printed, committed, persisted in project configuration or sent as review content. Jev's
initial diff screening flagged `runner.py` and `results.py` for inspection. These scores
prioritized review; direct source inspection and failing regressions established the defects.

The follow-up diff screen still marked persistence/setup code high-risk and reported zero
test files despite the changed `tests_mlx/test_*.py` unittest files. That `needs-test`
heuristic was checked against actual discovery and passing regressions, not accepted blindly.
A focused untruncated Jev pass also recognized the recovery/validation tests. The risk flags
remain reasons for independent review, not evidence that the code is now certified.
Changes are split into result-persistence, numerical, conversion, runner and documentation
commits to make that review easier.

## Reproduced defects fixed

1. **Missing completion summary after a crash.** Receipt-only resume returned success without
   recreating `summary.json`. It now reconstructs the report from verified receipts without
   loading models. It does not fabricate a peak-memory value for a no-model invocation.
2. **Input could overlap generated output.** An input named `summary.json` inside the result
   directory could be overwritten by the final report. Preflight now requires inputs outside
   the output directory, before even a dry-run is accepted.
3. **Malformed receipt shape handling.** Non-object envelopes raised an unstructured
   AttributeError; null results collided with the missing-receipt sentinel. Readers reject
   non-object envelopes/results, and writers require result objects.
4. **NaN/Inf could become a successful answer.** Last hidden states, link results after
   destination-dtype conversion, and solver logits now fail closed on non-finite values.
   Tests include finite FP32 link parameters overflowing during an FP16 cast.
5. **Allocator setup could leak process state.** A cache-limit failure after changing the
   memory limit bypassed the old cleanup block. An ExitStack now registers each restoration
   as the resource is changed, including signal handlers and allocator/cache settings.
6. **New results directories were not private by default.** Under umask 022 they were 0755.
   New directories are now created with 0700. Existing directory permissions are not modified.
7. **Rejected pickled adapters escaped structured CLI errors.** Tensor-only PyTorch loader
   errors are translated to validation errors. There is no fallback to unrestricted pickle
   and no added deserialization allowlist.

Each behavioral correction above was preceded by an executed failing regression. Jev triaged
those failures, but fixes were selected by inspecting the actual code and traceback.

## Expanded evidence

- Real child processes are SIGKILLed just before and after receipt rename, using explicit
  synchronization rather than timing guesses. Prior receipts remain readable; the result
  directory lock can be reacquired only after the child dies; orphan staging files are ignored.
- A real worker receives SIGTERM at a controlled loading boundary. It exits with a structured
  cancellation error, publishes no sample/summary, and releases its lock.
- Rejected adapter cases cover missing/extra keys, non-finite and integer tensors, nested
  state dictionaries, pickled modules, wrong shapes and existing destinations.
- Original tiny same-weight PyTorch decoder/link comparisons and recurrent native smoke remain
  in the full suite, with unchanged numerical tolerances and TF32 disabled.

## Remaining gates and cautions

- These are targeted **process-crash** tests, not physical power-loss certification or proof
  that a hung Metal kernel can be interrupted promptly.
- Allocation limits and stage checks are cooperative. No hard OS resource sandbox is supplied.
- New numerical checks introduce synchronization; their production-size overhead is unmeasured.
- Model directories must remain immutable during a run. Hash verification is not isolation
  against a malicious concurrent writer.
- The receipt identity includes source hashes: this hardening changes identity, so results
  from the earlier code require a new output directory rather than mixed-version resume.
- No independent subagent/human code review is claimed. Jev screening does not replace it.
- No merge, release, model upload, production/mini/CI changes or background training occurred.

## Next operator decisions

1. Approve an exact compatible model/link bundle and its acquisition/resource budget.
2. Approve a reference-quality dataset, scoring method and full-decoder tolerance protocol.
3. Obtain independent code review before merge or release.

Until then, keep the PR draft. The safe immediate command remains
`python -m recursive_mlx smoke`; see [runbook](MLX_GUIDE.md) and
[verification record](MLX_VERIFICATION.md) for reproducible checks.
