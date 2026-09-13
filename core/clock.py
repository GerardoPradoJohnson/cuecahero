from dataclasses import dataclass
import math

@dataclass
class SimulationClock:
    game_step: float = 1 / 30
    neural_step_ms: float = 10.
    speed: float = 1.
    game_seconds: float = 0.
    neural_ms: float = 0.
    ticks: int = 0

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in (self.game_step, self.neural_step_ms, self.speed)):
            raise ValueError('Clock intervals must be finite and positive')
        if not math.isclose(self.neural_step_ms / .1, round(self.neural_step_ms / .1), abs_tol=1e-8):
            raise ValueError('Neural interval must be a multiple of 0.1 ms')

    def advance(self, neural: bool = True):
        self.ticks += 1
        self.game_seconds = self.ticks * self.game_step
        if neural:
            self.neural_ms += self.neural_step_ms

    def reset(self):
        self.game_seconds = self.neural_ms = 0.
        self.ticks = 0
