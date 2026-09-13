"""Fixed supervised BCI calibration on postsynaptic interneuron spikes.

Calibration labels are used OFFLINE only. Runtime takes NeuralActivity only.
This is decoder calibration, never a claim of plasticity inside MaleCNS.
"""
import json
import math
from collections import deque
from pathlib import Path
import numpy as np
from core.contracts import Action

class CalibratedDecoder:
    def __init__(self, model):
        record=json.loads(Path(model).read_text()) if isinstance(model,(str,Path)) else model
        if record.get('schema')!=1:
            raise ValueError('Unknown readout calibration schema')
        self.record=record
        self.groups=[np.asarray(g,dtype=np.int32) for g in record.get('groups',[[i] for i in record['indices']])]
        self.indices=np.unique(np.concatenate(self.groups))
        self.feature_count=len(self.groups)
        self.lags=record.get('lags',[0])
        if not self.lags or self.lags[0]!=0 or any(not isinstance(x,int) or x<0 or x>100 for x in self.lags):
            raise ValueError('Invalid neural history lags')
        dimensions=self.feature_count*len(self.lags)
        self.mean=np.asarray(record['mean'],dtype=np.float64)
        self.scale=np.asarray(record['scale'],dtype=np.float64)
        self.weights=np.asarray(record['weights'],dtype=np.float64)
        self.bias=np.asarray(record['bias'],dtype=np.float64)
        if self.weights.shape!=(dimensions,4) or self.mean.shape!=(dimensions,) or self.scale.shape!=(dimensions,) or self.bias.shape!=(4,) or np.any(self.scale<=0) or any(not np.isfinite(a).all() for a in (self.weights,self.mean,self.scale,self.bias)):
            raise ValueError('Invalid calibration arrays')
        self.threshold=record.get('threshold',.45)
        self.release=record.get('release',.2)
        self.tau_ms=record['tau_ms']
        self.cooldown_ms=record.get('cooldown_ms',30.)
        if not np.isfinite([self.threshold,self.release,self.tau_ms,self.cooldown_ms]).all() or not 0 <= self.release < self.threshold <= 1 or self.tau_ms <= 0 or self.cooldown_ms < 0:
            raise ValueError('Invalid calibration timing or thresholds')
        if any(g.ndim != 1 or len(g) == 0 or np.any(g < 0) for g in self.groups):
            raise ValueError('Invalid neuronal population')
        if not np.array_equal(self.indices, np.unique(record['indices'])):
            raise ValueError('Population indices do not match calibration provenance')
        self.reset()

    def reset(self):
        self.filtered=np.zeros(self.feature_count,float)
        self.history=deque([self.filtered.copy() for _ in range(max(self.lags)+1)],maxlen=max(self.lags)+1)
        self.rates=np.zeros(4)
        self.armed=np.ones(4,bool)
        self.last=np.full(4,-1e12)

    def decode(self, activity):
        if activity.duration_ms<=0 or not math.isfinite(activity.duration_ms) or np.any(self.indices<0) or np.any(self.indices>=len(activity.counts)):
            raise ValueError('Invalid activity for calibrated decoder')
        alpha=1-math.exp(-activity.duration_ms/self.tau_ms)
        self.filtered+=alpha*(np.array([activity.counts[g].mean() for g in self.groups])*1000/activity.duration_ms-self.filtered)
        self.history.appendleft(self.filtered.copy())
        features=np.concatenate([self.history[lag] for lag in self.lags])
        self.rates=np.clip(((features-self.mean)/self.scale)@self.weights+self.bias,0,1)
        self.armed |= self.rates<self.release
        fire=(self.rates>=self.threshold)&self.armed&(activity.timestamp_ms-self.last>=self.cooldown_ms)
        self.last[fire]=activity.timestamp_ms
        self.armed[fire]=False
        return Action(tuple(float(x) for x in fire))
