"""Dependency-free bounded execution contracts."""

from dataclasses import dataclass
import math
import time


class Cancelled(RuntimeError):
    """Cooperative cancellation; process exit remains the owner’s responsibility."""


@dataclass(frozen=True)
class Limits:
    rounds: int = 1
    latent_steps: int = 4
    max_tokens: int = 32
    max_context: int = 2048
    max_samples: int = 1
    timeout_s: float = 120.0

    def __post_init__(self):
        for name, maximum in (
            ("rounds", 16),
            ("latent_steps", 256),
            ("max_tokens", 4096),
            ("max_context", 32768),
            ("max_samples", 10000),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in 1..{maximum}")
        if (
            isinstance(self.timeout_s, bool)
            or not math.isfinite(self.timeout_s)
            or self.timeout_s <= 0
        ):
            raise ValueError("timeout_s must be finite and positive")


class Budget:
    def __init__(self, limits, clock=time.monotonic, cancelled=lambda: False):
        self.limits, self.clock, self.cancelled = limits, clock, cancelled
        self.started = clock()

    def check(self, context=0):
        if self.cancelled():
            raise Cancelled("cancellation requested")
        if self.clock() - self.started >= self.limits.timeout_s:
            raise TimeoutError("execution deadline exceeded")
        if context > self.limits.max_context:
            raise ValueError("context budget exceeded; input was not truncated")
