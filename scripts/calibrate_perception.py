"""Offline calibration of a fixed readout, using only real neural spike features.

The full MaleCNS graph is simulated. Labels describe controlled moving calibration
stimuli; they never enter the runtime encoder or decoder. Saved weights belong to
the external decoder, NOT the biological connectome.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pyarrow.feather as feather
from scipy.linalg import solve
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.paths import ROOT,DATA,GRAPH
from core.malecns import MaleCNSCore
from core.contracts import Action
from environments import CuecaHeroEnvironment
from sensors.contrast import ContrastVisualEncoder
from decoders.calibrated import CalibratedDecoder
from decoders.populations import lamina_populations

SENSOR={'viewport':[.08,.10,.92,.30],'black_level':.15,'white_level':.8,'gain':120.,'lamina_bias':12.,'pooling':[.04,.0125],'tau_ms':10.}

def trial(brain, encoder, lane, onset, duration=3.6, step=1/30):
    env=CuecaHeroEnvironment();env.notes=[] if lane<0 else [{'id':0,'lane':lane,'at':onset,'judgement':None}]
    env.duration=duration
    brain.reset();encoder.reset()
    for tick in range(round(duration/step)):
        activity=brain.advance(encoder.encode(env.get_observation(),10),10)
        label=np.zeros(4)
        if lane>=0 and abs((env.time+step)-onset)<=.10:
            label[lane]=1
        yield activity,label,env
        env.step(Action(),step)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'outputs/calibration/perception-v2');args=parser.parse_args()
    if args.output.exists():parser.error('Output exists; choose a new directory')
    args.output.mkdir(parents=True)
    audit=json.loads((DATA/'outputs/doom/audit/data-integrity.json').read_text())
    h=hashlib.sha256()
    with GRAPH.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    if not audit['passed'] or h.hexdigest()!=audit['graph_sha256']:raise ValueError('Graph audit mismatch')
    brain=MaleCNSCore();encoder=ContrastVisualEncoder(brain.state.retina,brain.state.uv,brain.state.lamina,**SENSOR)
    nodes=feather.read_table(DATA/'connectome_data/malecns_v1/normalized/neurons.feather',columns=['cell_type','source_id']).to_pandas()
    groups,group_labels=lamina_populations(brain.state,encoder,nodes.cell_type.to_numpy())
    candidates=np.unique(np.concatenate(groups))
    print(f'{len(groups)} graph-supported pools, {len(candidates)} interneurons',flush=True)
    features=[];labels=[];tau=20.;start=time.perf_counter()
    for n,(lane,onset) in enumerate([(lane,onset) for onset in [1.1,1.57,2.31,2.79] for lane in [-1,0,1,2,3]]):
        filtered=np.zeros(len(groups))
        for activity,label,env in trial(brain,encoder,lane,onset):
            filtered+=(1-np.exp(-10/tau))*(np.array([activity.counts[g].mean() for g in groups])*100-filtered)
            features.append(filtered.astype(np.float32).copy());labels.append(label)
        print(f'Calibration {n+1}/20: lane={lane}, onset={onset}',flush=True)
    X=np.asarray(features);Y=np.asarray(labels)
    selected=candidates
    X=X.astype(float);mean=X.mean(0);scale=np.maximum(X.std(0),1.)
    Z=(X-mean)/scale
    Z=np.c_[Z,np.ones(len(Z))]
    importance=np.where(Y.sum(1)>0,5.,1.)
    ridge=np.eye(Z.shape[1])*60;ridge[-1,-1]=.1
    weights=solve(Z.T@(Z*importance[:,None])+ridge,Z.T@(Y*importance[:,None]),assume_a='pos')
    record={'schema':1,'kind':'supervised_fixed_bci_calibration','indices':selected.tolist(),'groups':groups,'group_labels':group_labels,
        'source_ids':[str(x) for x in nodes.source_id.iloc[selected]],'cell_types':nodes.cell_type.iloc[selected].tolist(),
        'mean':mean.tolist(),'scale':scale.tolist(),'weights':weights[:-1].tolist(),'bias':weights[-1].tolist(),
        'tau_ms':tau,'threshold':.35,'release':.20,'cooldown_ms':30.,'sensor':SENSOR,
        'graph_sha256':h.hexdigest(),'training':'20 controlled moving-stimulus trials; four onset times; graph-supported lamina pools; no gameplay policy or future-note input at runtime',
        'limits':'External readout calibration. No connectome weights were trained. Not biological learning.'}
    (args.output/'model.json').write_text(json.dumps(record,indent=2)+'\n')
    np.savez_compressed(args.output/'training.npz',features=X,labels=Y,indices=selected)
    validations=[]
    for lane,onset in [(lane,onset) for onset in [1.81,2.64] for lane in [-1,0,1,2,3]]:
        decoder=CalibratedDecoder(record);presses=[];scores=[]
        for activity,label,env in trial(brain,encoder,lane,onset):
            action=decoder.decode(activity)
            for index,value in enumerate(action.values):
                if value:presses.append({'lane':index,'time':round(env.time+1/30,5),'error_ms':round((env.time+1/30-onset)*1000,2)})
            scores.append(decoder.rates.tolist())
        result={'lane':lane,'onset':onset,'presses':presses,'max_scores':np.max(scores,axis=0).tolist()}
        validations.append(result);print('HELD OUT '+json.dumps(result),flush=True)
    (args.output/'validation.json').write_text(json.dumps({'trials':validations,'wall_seconds':time.perf_counter()-start,'selected_neurons':len(selected)},indent=2)+'\n')
if __name__=='__main__':main()
