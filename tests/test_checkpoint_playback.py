import hashlib
import json
import numpy as np
import pytest
from training import checkpoints
from training.rl import ReinforcementReadoutTrainer


def test_sigmoid_checkpoint_keeps_keys_down_and_resets():
    model=dict(schema=1, indices=[0], groups=[[0]], mean=[0],scale=[1],
               weights=[[1,0,0,0]],bias=[0,-10,-10,-10],tau_ms=20,
               threshold=.55,release=.4,action_mode='held_sigmoid')
    trainer=ReinforcementReadoutTrainer(model)
    assert trainer.step(np.array([2.]),0,explore=False)[0].values==(1.,0.,0.,0.)
    assert trainer.step(np.array([2.]),33,explore=False)[0].values==(1.,0.,0.,0.)
    assert trainer.step(np.array([-2.]),66,explore=False)[0].values==(0.,0.,0.,0.)
    with pytest.raises(ValueError):trainer.step(np.array([2.]),99,explore=True)
    trainer.reset_runtime()
    assert not trainer.held_actions.any()


def test_recording_checks_provenance_and_rejects_paths(tmp_path,monkeypatch):
    monkeypatch.setattr(checkpoints,'ROOT',tmp_path)
    folder=tmp_path/'outputs/training/run';folder.mkdir(parents=True)
    files={'checkpoint':folder/'episode-0200.npz','playback':folder/'episode-0200.playback.json.gz',
           'song':tmp_path/'config/songs/la_consentida.json','audio':tmp_path/'frontend/audio/consentida.mp3'}
    metadata={}
    for key,path in files.items():
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(key.encode())
        metadata[key+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    metadata['checkpoint']=files['checkpoint'].name
    metadata['playback']=files['playback'].name
    (folder/'evaluation.json').write_text(json.dumps(metadata))
    relative=str(files['checkpoint'].relative_to(tmp_path))
    assert checkpoints.recording_path(relative)==files['playback']
    files['audio'].write_bytes(b'changed audio')
    with pytest.raises(ValueError):checkpoints.recording_path(relative)
    with pytest.raises(ValueError):checkpoints.resolve_checkpoint('../outside.npz')
    with pytest.raises(ValueError):checkpoints.resolve_checkpoint('config/songs/la_consentida.json')
