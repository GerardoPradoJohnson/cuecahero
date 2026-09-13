"""Synthetic fixtures for learner persistence, split isolation and determinism."""
import hashlib
import json
import numpy as np
import pytest
from training.readout import EpisodicReadoutTrainer, temporal_features


def fixture(tmp_path):
    rng=np.random.default_rng(123)
    model={'schema':1,'indices':[0,1,2,3],'mean':[0]*8,'scale':[1]*8,
           'weights':np.zeros((8,4)).tolist(),'bias':[0]*4,'tau_ms':20,
           'lags':[0,2],'training_files_sha256':{}}
    paths=[]
    for i in range(3):
        raw=rng.random((30,4));labels=(raw>.75).astype(float)
        path=tmp_path/f'train-{i}.npz';np.savez(path,features=raw,labels=labels)
        model['training_files_sha256'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        paths.append(path)
    return model,paths


def test_resume_matches_uninterrupted_and_erasure_removes_fit(tmp_path):
    model,paths=fixture(tmp_path)
    full=EpisodicReadoutTrainer(model)
    full.consume(paths[0]);checkpoint=tmp_path/'checkpoint.npz';full.save(checkpoint)
    restored=EpisodicReadoutTrainer.load(checkpoint)
    for path in paths[1:]:
        full.consume(path);restored.consume(path)
    np.testing.assert_array_equal(full.gram,restored.gram)
    np.testing.assert_array_equal(full.rhs,restored.rhs)
    assert full.model()==restored.model()
    assert np.any(full.model()['weights'])
    assert not np.any(EpisodicReadoutTrainer(model).model()['weights'])
    with pytest.raises(FileExistsError):full.save(checkpoint)


def test_rejects_changed_or_heldout_episode_without_mutation(tmp_path):
    model,paths=fixture(tmp_path);trainer=EpisodicReadoutTrainer(model)
    before=trainer.gram.copy()
    with paths[0].open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='checksum'):trainer.consume(paths[0])
    heldout=tmp_path/'heldout.npz';heldout.write_bytes(paths[1].read_bytes())
    with pytest.raises(ValueError,match='split'):trainer.consume(heldout)
    np.testing.assert_array_equal(before,trainer.gram)
    assert trainer.episodes==[]


def test_checkpoint_detects_tampering(tmp_path):
    model,paths=fixture(tmp_path);trainer=EpisodicReadoutTrainer(model)
    trainer.consume(paths[0]);path=tmp_path/'checkpoint.npz';trainer.save(path)
    with np.load(path) as data:arrays={key:data[key] for key in data.files}
    arrays['rhs'][0,0]+=1
    np.savez(path,**arrays)
    with pytest.raises(ValueError,match='checksum'):EpisodicReadoutTrainer.load(path)


def test_episode_history_never_wraps_from_previous_episode():
    raw=np.arange(12).reshape(3,4)
    features=temporal_features(raw,[0,2,8])
    np.testing.assert_array_equal(features[:2,4:8],0)
    np.testing.assert_array_equal(features[2,4:8],raw[0])
    np.testing.assert_array_equal(features[:,8:],0)
