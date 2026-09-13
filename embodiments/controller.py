import math
from core.contracts import Action

class GameController:
    def apply(self, action):
        if len(action.values) != 4 or not all(math.isfinite(x) for x in action.values):
            raise ValueError('GameController requires four finite actions')
        return Action(tuple(float(x >= .5) for x in action.values))
