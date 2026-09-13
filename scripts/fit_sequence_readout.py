"""Fit a fixed temporal BCI using saved training spikes only; no re-simulation."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.linalg import solve

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Output exists; choose a new directory')
    model=json.loads((args.calibration/'model.json').read_text())
    lags=[0,2,4,8];X=[];Y=[];sources={}
    for path in sorted(args.calibration.glob('training-*.npz')):
        sources[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as data:
            raw=data['features'];delayed=[]
            for lag in lags:
                shifted=np.zeros_like(raw)
                shifted[lag:]=raw[:len(raw)-lag]
                delayed.append(shifted)
            X.append(np.concatenate(delayed,axis=1));Y.append(data['labels'])
    if len(X)!=6:raise ValueError('Expected six complete training charts')
    X=np.concatenate(X);Y=np.concatenate(Y)
    mean=X.mean(0);scale=np.maximum(X.std(0),1.)
    Z=np.c_[(X-mean)/scale,np.ones(len(X))];importance=np.where(Y.sum(1)>0,5.,1.)
    ridge=np.eye(Z.shape[1])*60;ridge[-1,-1]=.1
    weights=solve(Z.T@(Z*importance[:,None])+ridge,Z.T@(Y*importance[:,None]),assume_a='pos')
    model.update(lags=lags,mean=mean.tolist(),scale=scale.tolist(),weights=weights[:-1].tolist(),bias=weights[-1].tolist(),threshold=.5,release=.25)
    model['training']+=' Temporal taps use the current and 2/4/8 preceding neural observations.'
    model['training_files_sha256']=sources
    args.output.mkdir(parents=True)
    (args.output/'model.json').write_text(json.dumps(model,indent=2)+'\n')

if __name__=='__main__':main()
