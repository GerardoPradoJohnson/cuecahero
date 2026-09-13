"""Calibrate an external BCI on real neural responses to full training charts.

Teacher labels/actions exist only in this offline collection. Runtime inference
receives only spikes. No MaleCNS connections or weights are changed.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy.linalg import solve
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.paths import GRAPH
from core.malecns import MaleCNSCore
from core.contracts import Action
from environments import CuecaHeroEnvironment
from sensors.contrast import ContrastVisualEncoder


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base-model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Output exists; choose a new directory')
    args.output.mkdir(parents=True)
    model=json.loads(args.base_model.read_text())
    digest=hashlib.sha256()
    with GRAPH.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    if digest.hexdigest()!=model['graph_sha256']:raise ValueError('Graph mismatch')
    brain=MaleCNSCore()
    encoder=ContrastVisualEncoder(brain.state.retina,brain.state.uv,brain.state.lamina,**model['sensor'])
    groups=[np.asarray(g) for g in model['groups']]
    X=[];Y=[];started=time.perf_counter()
    for seed,teacher in [(211,True),(307,True),(401,True),(503,True),(601,False),(701,False)]:
        env=CuecaHeroEnvironment();env.reset(seed);brain.reset();encoder.reset()
        filtered=np.zeros(len(groups));features=[];labels=[]
        while not env.is_done():
            activity=brain.advance(encoder.encode(env.get_observation(),10),10)
            filtered+=(1-np.exp(-10/model['tau_ms']))*(np.array([activity.counts[g].mean() for g in groups])*100-filtered)
            target=np.zeros(4);action=np.zeros(4)
            for note in env.notes:
                error=env.time+1/30-note['at']
                if note['judgement'] is None:
                    if -.10<=error<=.10:target[note['lane']]=1
                    if 0<=error<1/30+1e-9:action[note['lane']]=1
            features.append(filtered.copy());labels.append(target)
            env.step(Action(tuple(action)) if teacher else Action(),1/30)
        np.savez_compressed(args.output/f'training-{seed}.npz',features=features,labels=labels)
        X.extend(features);Y.extend(labels)
        print(f'Collected seed={seed}, teacher={teacher}, frames={len(features)}',flush=True)
    X=np.asarray(X);Y=np.asarray(Y);mean=X.mean(0);scale=np.maximum(X.std(0),1.)
    Z=np.c_[(X-mean)/scale,np.ones(len(X))];importance=np.where(Y.sum(1)>0,5.,1.)
    ridge=np.eye(Z.shape[1])*60;ridge[-1,-1]=.1
    weights=solve(Z.T@(Z*importance[:,None])+ridge,Z.T@(Y*importance[:,None]),assume_a='pos')
    model.update(mean=mean.tolist(),scale=scale.tolist(),weights=weights[:-1].tolist(),bias=weights[-1].tolist(),threshold=.5,release=.25)
    model.pop('threshold_selection',None)
    model['training']='Six full charts, seeds 211/307/401/503 (offline teacher actions) and 601/701 (no actions). Labels and teacher never enter runtime.'
    model['training_wall_seconds']=time.perf_counter()-started
    (args.output/'model.json').write_text(json.dumps(model,indent=2)+'\n')
    print('Saved frozen sequence readout',flush=True)

if __name__=='__main__':main()
