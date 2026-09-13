"""DOOMFLY's declared R1–R6 luminance proxy, outside the neural engine."""
import math
import numpy as np
from core.contracts import NeuralStimulus

class VisualEncoder:
    def __init__(self, retina, uv, lamina, lamina_bias=12.):
        self.retina, self.uv, self.lamina = retina, uv, lamina
        self.lamina_bias = lamina_bias
        self.reset()

    def reset(self):
        self.filtered = np.zeros(len(self.retina), dtype=np.float32)

    def encode(self, observation, duration_ms):
        image = observation.rgb
        h, w, channels = image.shape
        if channels != 3 or not math.isfinite(duration_ms) or duration_ms <= 0 or image.dtype != np.uint8:
            raise ValueError('Finite RGB observation required')
        x = np.clip(np.rint(self.uv[:,0] * (w - 1)).astype(int), 0, w - 1)
        y = np.clip(np.rint(self.uv[:,1] * (h - 1)).astype(int), 0, h - 1)
        luminance = image[y,x].astype(np.float32) @ np.array([.2126,.7152,.0722], np.float32) / 255.
        self.filtered += (1 - math.exp(-duration_ms / 10)) * (luminance - self.filtered)
        currents = 30 * self.filtered / (.02 + self.filtered)
        return NeuralStimulus(np.concatenate((self.lamina, self.retina)),
                              np.concatenate((np.full(len(self.lamina), self.lamina_bias, np.float32), currents)))
