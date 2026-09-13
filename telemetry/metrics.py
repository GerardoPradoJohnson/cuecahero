from collections import deque

class Metrics:
    def __init__(self):
        self.history = deque(maxlen=180)
        self.total_spikes = 0

    def record(self, activity, wall_seconds, game_seconds):
        spikes = int(activity.counts.sum()) if activity else 0
        self.total_spikes += spikes
        value = {'spikes':spikes, 'total_spikes':self.total_spikes,
                 'active_neurons':int((activity.counts > 0).sum()) if activity else 0,
                 'step_wall_ms':round(wall_seconds * 1000, 3),
                 'game_seconds':round(game_seconds, 6),
                 'neural_ms':round(activity.timestamp_ms, 3) if activity else 0.}
        self.history.append(value)
        return value
