"""Uncached, batch-one sequential RecursiveMAS with native latent exchange."""

import time
import mlx.core as mx
from .contracts import Budget, Limits
from .links import InnerLink, OuterLink, apply_link


def latent_rollout(backend, inner, embeds, steps, budget):
    if type(steps) is not int or steps < 1:
        raise ValueError("latent steps must be positive")
    budget.check(embeds.shape[1] + steps)
    states = []
    for _ in range(steps):
        budget.check(embeds.shape[1])
        last = backend.hidden(embeds)[:, -1:, :]
        mx.eval(last)
        states.append(last)
        next_embed = apply_link(inner, last, embeds.dtype)
        embeds = mx.concatenate([embeds, next_embed], axis=1)
        mx.eval(embeds)
    return mx.concatenate(states, axis=1)


def _chat(tokenizer, prompt, tokenize):
    from inference.prompts import SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    kwargs = dict(tokenize=tokenize, add_generation_prompt=True, enable_thinking=False)
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


def prompt_embeds(backend, role, question, incoming):
    from inference.prompts import (
        PLANNER_SLOT,
        REFINED_SLOT,
        FEEDBACK_SLOT,
        build_math_planner_prompt,
        build_math_planner_prompt_with_feedback_slot,
        build_math_refiner_prompt_with_slot,
        build_math_solver_prompt_with_slots,
    )

    if any(slot in question for slot in (PLANNER_SLOT, REFINED_SLOT, FEEDBACK_SLOT)):
        raise ValueError("question contains a reserved latent-slot marker")
    if role == "planner" and incoming is None:
        ids = _chat(backend.tokenizer, build_math_planner_prompt(question), True)
        if not isinstance(ids, list) or not all(type(t) is int for t in ids):
            raise ValueError("unsupported chat-template token result")
        return backend.embed(ids)
    if role == "planner":
        prompt, slot = (
            build_math_planner_prompt_with_feedback_slot(question),
            FEEDBACK_SLOT,
        )
    elif role == "critic":
        prompt, slot = build_math_refiner_prompt_with_slot(question), PLANNER_SLOT
    else:
        prompt, slot = build_math_solver_prompt_with_slots(question), REFINED_SLOT
    rendered = _chat(backend.tokenizer, prompt, False)
    if not isinstance(rendered, str) or rendered.count(slot) != 1 or incoming is None:
        raise ValueError("chat template must preserve exactly one latent slot")
    prefix, suffix = rendered.split(slot)
    left = backend.embed(backend.tokenizer.encode(prefix, add_special_tokens=False))
    right = backend.embed(backend.tokenizer.encode(suffix, add_special_tokens=False))
    return mx.concatenate([left, incoming.astype(left.dtype), right], axis=1)


def run_sequential(
    roles,
    inner,
    outer,
    question,
    budget,
    emit=lambda event: None,
    build_prompt=prompt_embeds,
):
    """Greedy-only reference path. All latent tensors stay within this worker."""
    started, stages = time.monotonic(), []
    feedback = None

    def stage(role, incoming, edge, label):
        t = time.monotonic()
        backend = roles[role]
        embeds = build_prompt(backend, role, question, incoming)
        raw = latent_rollout(
            backend, inner[role], embeds, budget.limits.latent_steps, budget
        )
        own = apply_link(inner[role], raw, embeds.dtype)
        # Reference solver feedback stores float32, other edges preserve source embedding dtype.
        dtype = mx.float32 if label == "solver_feedback" else embeds.dtype
        mapped = apply_link(outer[edge], own, dtype)
        mx.eval(mapped)
        event = {
            "type": "stage_completed",
            "role": label,
            "round": round_index + 1,
            "latent_steps": budget.limits.latent_steps,
            "elapsed_s": time.monotonic() - t,
        }
        stages.append(event)
        emit(event)
        return mapped

    for round_index in range(budget.limits.rounds):
        plan = stage("planner", feedback, "outer_12", "planner")
        refined = stage("critic", plan, "outer_23", "critic")
        if round_index + 1 < budget.limits.rounds:
            feedback = stage("solver", refined, "outer_31", "solver_feedback")
    backend = roles["solver"]
    embeds = build_prompt(backend, "solver", question, refined)
    budget.check(embeds.shape[1] + budget.limits.max_tokens)
    eos = getattr(backend.tokenizer, "eos_token_id", None)
    eos_ids = set(getattr(backend.tokenizer, "eos_token_ids", []) or [])
    eos_ids.update(eos if isinstance(eos, list) else [eos] if eos is not None else [])
    tokens, t = [], time.monotonic()
    for _ in range(budget.limits.max_tokens):
        budget.check(embeds.shape[1])
        logits = backend.logits(backend.hidden(embeds)[:, -1:, :])
        token = int(mx.argmax(logits[0, -1]).item())
        tokens.append(token)
        if token in eos_ids:
            break
        embeds = mx.concatenate([embeds, backend.embed([token])], axis=1)
        mx.eval(embeds)
    budget.check()
    event = {
        "type": "stage_completed",
        "role": "solver",
        "round": budget.limits.rounds,
        "output_tokens": len(tokens),
        "elapsed_s": time.monotonic() - t,
    }
    stages.append(event)
    emit(event)
    text = (
        backend.tokenizer.decode(tokens, skip_special_tokens=True)
        if backend.tokenizer
        else None
    )
    return {
        "backend": "mlx",
        "rounds": budget.limits.rounds,
        "token_ids": tokens,
        "text": text,
        "stages": stages,
        "elapsed_s": time.monotonic() - started,
    }


def smoke():
    """Tiny random MLX-LM models: proves execution plumbing, never answer quality."""
    from .backend import tiny_backend

    mx.random.seed(7)
    roles = {
        r: tiny_backend(f, 8)
        for r, f in (("planner", "qwen3"), ("critic", "llama"), ("solver", "qwen2"))
    }
    inner = {r: InnerLink(8) for r in roles}
    outer = {e: OuterLink(8, 8) for e in ("outer_12", "outer_23", "outer_31")}

    def synthetic_prompt(backend, role, question, incoming):
        prefix, suffix = backend.embed([1, 2]), backend.embed([3])
        return (
            mx.concatenate([prefix, incoming.astype(prefix.dtype), suffix], axis=1)
            if incoming is not None
            else prefix
        )

    result = run_sequential(
        roles,
        inner,
        outer,
        "synthetic",
        Budget(Limits(rounds=2, latent_steps=2, max_tokens=3, max_context=64)),
        build_prompt=synthetic_prompt,
    )
    result["qualification"] = "synthetic_only"
    result["peak_mlx_bytes"] = mx.get_peak_memory()
    return result
