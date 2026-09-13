from dataclasses import dataclass
from typing import Protocol, Mapping, Any, runtime_checkable
import numpy as np

@dataclass(frozen=True)
class Observation:
    rgb: np.ndarray
    timestamp: float

@dataclass(frozen=True)
class MultimodalObservation(Observation):
    audio: np.ndarray | None = None
    proprioception: Mapping[str, float] | None = None

@dataclass(frozen=True)
class NeuralStimulus:
    indices: np.ndarray
    currents: np.ndarray

@dataclass(frozen=True)
class NeuralActivity:
    counts: np.ndarray
    duration_ms: float
    timestamp_ms: float

@dataclass(frozen=True)
class Action:
    values: tuple[float, ...] = (0., 0., 0., 0.)

@dataclass(frozen=True)
class RewardSignal:
    magnitude: float
    source: str
    timestamp: float

@runtime_checkable
class Environment(Protocol):
    def reset(self, seed: int = 0) -> Mapping[str, Any]: ...
    def step(self, action: Action, dt: float) -> Mapping[str, Any]: ...
    def get_observation(self) -> Observation: ...
    def get_reward(self) -> RewardSignal: ...
    def is_done(self) -> bool: ...
    def get_state(self) -> Mapping[str, Any]: ...

class SensorEncoder(Protocol):
    def encode(self, observation: Observation, duration_ms: float) -> NeuralStimulus: ...
    def reset(self) -> None: ...

class NeuralEngine(Protocol):
    def advance(self, stimulus: NeuralStimulus, duration_ms: float) -> NeuralActivity: ...
    def reset(self) -> None: ...

class NeuralDecoder(Protocol):
    def decode(self, activity: NeuralActivity) -> Action: ...
    def reset(self) -> None: ...

class Embodiment(Protocol):
    def apply(self, action: Action) -> Action: ...

class PlasticityModel(Protocol):
    def apply(self, neural_state: NeuralEngine, reward: RewardSignal) -> None: ...
