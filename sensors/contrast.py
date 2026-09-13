"""Explicit engineering viewport + receptive-field pooling, not measured eye pose.

Only RGB pixels enter. No lanes, rewards, note times or actions are inspected.
Original neuron IDs and graph connections are unchanged.
"""
import math
import numpy as np
from scipy.ndimage import gaussian_filter
from core.contracts import NeuralStimulus

class ContrastVisualEncoder:
    def __init__(self, retina, uv, lamina, viewport=(.08,.10,.92,.30), black_level=.15,
                 white_level=.8, gain=120., lamina_bias=12., pooling=(.04,.0125), tau_ms=10.):
        rectangle=np.asarray(viewport,dtype=float)
        if rectangle.shape!=(4,) or not np.isfinite(rectangle).all() or np.any(rectangle[:2]<0) or np.any(rectangle[2:]>1) or np.any(rectangle[2:]<=rectangle[:2]):
            raise ValueError('Invalid normalized viewport')
        if not 0<=black_level<white_level<=1 or not 0<gain<=500 or tau_ms<=0 or not np.isfinite([gain,lamina_bias,tau_ms,*pooling]).all() or len(pooling)!=2 or min(pooling)<=0:
            raise ValueError('Invalid contrast parameters')
        uv=np.asarray(uv)
        included=((uv>=rectangle[:2])&(uv<=rectangle[2:])).all(axis=1)
        self.retina=np.asarray(retina)[included]
        self.uv=(uv[included]-rectangle[:2])/(rectangle[2:]-rectangle[:2])
        if not len(self.retina):
            raise ValueError('Viewport contains no official receptors')
        self.lamina=np.asarray(lamina)
        self.black_level,self.white_level,self.gain=black_level,white_level,gain
        self.pooling,self.tau_ms,self.lamina_bias=pooling,tau_ms,lamina_bias
        self.reset()

    def reset(self):
        self.filtered=np.zeros(len(self.retina),np.float32)

    def encode(self, observation, duration_ms):
        rgb=observation.rgb
        if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8 or not math.isfinite(duration_ms) or duration_ms<=0:
            raise ValueError('Finite duration and uint8 RGB required')
        h,w,_=rgb.shape
        luminance=rgb.astype(np.float32)@np.array([.2126,.7152,.0722],np.float32)/255
        contrast=np.clip((luminance-self.black_level)/(self.white_level-self.black_level),0,1)
        pooled=gaussian_filter(contrast,sigma=(h*self.pooling[0],w*self.pooling[1]),mode='nearest')
        x=np.rint(self.uv[:,0]*(w-1)).astype(int); y=np.rint(self.uv[:,1]*(h-1)).astype(int)
        sample=pooled[y,x]
        self.filtered+=(1-math.exp(-duration_ms/self.tau_ms))*(sample-self.filtered)
        return NeuralStimulus(np.concatenate((self.lamina,self.retina)),
            np.concatenate((np.full(len(self.lamina),self.lamina_bias,np.float32),self.gain*self.filtered)))
