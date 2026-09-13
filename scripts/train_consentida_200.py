"""200 supervised epochs of the external neural readout, then live evaluation.
Teachers label OFFLINE data only. Evaluation receives only real neural spikes.
"""
import copy,gzip,hashlib,json,math,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from threadpoolctl import threadpool_limits
from core.paths import ROOT
from core.contracts import Action,NeuralStimulus
from experiments.runtime import Experiment
from visualization.brain import load_geometry
from training.rl import ReinforcementReadoutTrainer

OUT=ROOT/'outputs/training/consentida-200-v2'
def write_json(path,data):
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(data,ensure_ascii=True,allow_nan=False),encoding='utf-8');tmp.replace(path)
def status(stage,**kw):
    write_json(OUT/'status.json',dict(stage=stage,**kw));print(json.dumps(dict(stage=stage,**kw)),flush=True)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def labels(env):
    t=env.time+1/30
    return np.array([float(any(n['lane']==lane and n['at']-.035<=t<=n['at']+max(.035,n.get('sustain',0)) for n in env.notes)) for lane in range(4)])
def neural(e):
    t=e.environment.time;phase=(t%(60/e.environment.bpm))/(60/e.environment.bpm)
    energy=float(np.clip(math.exp(-phase*9)+.5*math.exp(-((phase-.5)%1)*9),0,1))
    stimulus=e.encoder.encode(e.environment.get_observation(),10)
    ids=np.array(e.geometry['indices'])[e.geometry.get('auditory_indices',[])]
    if len(ids):stimulus=NeuralStimulus(np.r_[stimulus.indices,ids],np.r_[stimulus.currents,np.full(len(ids),18*energy+2,dtype=np.float32)])
    return e.brain.advance(stimulus,10)

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    threadpool_limits(1)
    status('initializing')
    cfg=json.loads((ROOT/'experiments/cueca_hero.json').read_text())
    geometry=load_geometry(cfg['visualization']['sample_size'])
    e=Experiment(cfg,geometry,sessions_dir=OUT/'sessions');e.close()
    e.load_brain();e.driver='neural';e.live_training_enabled=False
    song=json.loads((ROOT/'config/songs/la_consentida.json').read_text())
    e.environment.load_song(song)
    template=copy.deepcopy(e.decoder.record)
    X=[];Y=[]
    for teacher in (False,True):
        e.reset();e.trainer=ReinforcementReadoutTrainer(template)
        stage='collect_teacher' if teacher else 'collect_silent'
        started=time.perf_counter();frame=0
        while not e.environment.is_done():
            activity=neural(e)
            raw=e.trainer.extract_features(activity)*e.trainer.scale+e.trainer.mean
            target=labels(e.environment)
            X.append(raw.astype(np.float32));Y.append(target)
            e.environment.step(Action(tuple(target)) if teacher else Action(),1/30)
            frame+=1
            if frame%300==0:status(stage,frame=frame,seconds=round(e.environment.time,2),elapsed=round(time.perf_counter()-started,1),backend=e.brain.backend)
    X=np.asarray(X,dtype=np.float64);Y=np.asarray(Y)
    np.savez_compressed(OUT/'neural-examples.npz',features=X,targets=Y)
    mean=X.mean(0);scale=np.maximum(X.std(0),1.)
    Z=np.c_[(X-mean)/scale,np.ones(len(X))]
    W=np.zeros((Z.shape[1],4));W[-1]=-2.5
    m=np.zeros_like(W);v=m.copy();curve=[]
    weights=np.where(Y>.5,12.,1.)
    template.update(mean=mean.tolist(),scale=scale.tolist(),action_mode='held_sigmoid',
                    threshold=.55,release=.40,
                    training_method='Supervised offline neural readout: 200 complete dataset epochs; silent and teacher rollouts. Not 200 live brain generations.',
                    song_sha256=digest(ROOT/'config/songs/la_consentida.json'),audio_sha256=digest(ROOT/'frontend/audio/consentida.mp3'))
    trainer=ReinforcementReadoutTrainer(template)
    for epoch in range(1,201):
        logits=np.clip(Z@W,-30,30);pred=1/(1+np.exp(-logits))
        grad=Z.T@((pred-Y)*weights)/len(Z)+1e-4*W;grad[-1]-=1e-4*W[-1]
        m=.9*m+.1*grad;v=.999*v+.001*grad*grad
        W-=.015*(m/(1-.9**epoch))/(np.sqrt(v/(1-.999**epoch))+1e-8)
        loss=float(np.mean(weights*(np.logaddexp(0,logits)-Y*logits)))
        record=dict(index=epoch,generation=epoch,name=f'episode-{epoch:04d}',loss=loss,kind='supervised_epoch')
        trainer.episodes.append(record);curve.append(record)
        trainer.weights=W[:-1].copy();trainer.bias=W[-1].copy();trainer.opt_step=epoch
        trainer.m=-m.copy();trainer.v=v.copy()
        if epoch%25==0:
            checkpoint=OUT/f'episode-{epoch:04d}.npz';trainer.save(checkpoint)
            status('training',completed=epoch,target=200,loss=round(loss,5))
    write_json(OUT/'progress.json',dict(kind='supervised_neural_readout',episodes_completed=200,episodes=curve))
    status('evaluating',completed=200,target=200)
    e.trainer=ReinforcementReadoutTrainer.load(OUT/'episode-0200.npz')
    e.generation=200;e.reset();e.mode='running';e.publish()
    frames=[copy.deepcopy(e.snapshot)]
    started=time.perf_counter()
    while not e.environment.is_done():
        e.tick();frames.append(e.snapshot)
        if len(frames)%300==0:status('evaluating',completed=200,seconds=round(e.environment.time,2),elapsed=round(time.perf_counter()-started,1))
    result=e.environment.get_state()
    metrics={k:result[k] for k in ('score','hits','misses','wrong','accuracy','total_notes','best_combo')}
    checkpoint=OUT/'episode-0200.npz'
    playback=OUT/'episode-0200.playback.json.gz'
    playback_data=json.dumps(dict(header=dict(id='consentida-200-v2',kind='frozen_live_neural_evaluation',checkpoint_sha256=digest(checkpoint),song_sha256=template['song_sha256'],audio_sha256=template['audio_sha256']),frames=frames),ensure_ascii=True,allow_nan=False).encode()
    with gzip.open(playback,'wb',compresslevel=6) as stream:stream.write(playback_data)
    summary=dict(name='La Consentida - 200 epocas',generation=200,method=template['training_method'],checkpoint=checkpoint.name,playback=playback.name,metrics=metrics,checkpoint_sha256=digest(checkpoint),playback_sha256=digest(playback),song_sha256=template['song_sha256'],audio_sha256=template['audio_sha256'],backend=e.brain.backend)
    write_json(OUT/'evaluation.json',summary)
    status('complete',completed=200,target=200,metrics=metrics)

if __name__=='__main__':
    try:main()
    except Exception as err:
        if OUT.exists():status('failed',error=str(err))
        raise
