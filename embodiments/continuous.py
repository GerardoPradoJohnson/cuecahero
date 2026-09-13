"""Continuous embodiment applying physical momentum, deadband, and saturation limits."""
import math
from core.contracts import Action


class ContinuousEmbodiment:
    """Simulates physical inertia and actuator limits for continuous analog control."""

    def __init__(
        self,
        momentum: float = 0.3,
        deadzone: float = 0.04,
        limits: tuple[float, float] = (0.0, 1.0),
    ):
        if not 0.0 <= momentum < 1.0:
            raise ValueError('Momentum must be in [0.0, 1.0)')
        if not 0.0 <= deadzone < 0.5:
            raise ValueError('Deadzone must be in [0.0, 0.5)')
        self.momentum = momentum
        self.deadzone = deadzone
        self.min_val, self.max_val = limits
        self.current: list[float] = []

    def reset(self) -> None:
        self.current = []

    def apply(self, action: Action) -> Action:
        if not all(math.isfinite(x) for x in action.values):
            raise ValueError('Continuous actions must be finite numbers')

        if not self.current or len(self.current) != len(action.values):
            self.current = [0.0] * len(action.values)

        output = []
        for i, val in enumerate(action.values):
            # Deadzone suppression
            if abs(val) < self.deadzone:
                effective = 0.0
            else:
                effective = val

            # Physical momentum / inertia
            smoothed = self.current[i] * self.momentum + effective * (1.0 - self.momentum)
            clamped = max(self.min_val, min(self.max_val, smoothed))
            self.current[i] = clamped
            output.append(clamped)

        return Action(tuple(output))
