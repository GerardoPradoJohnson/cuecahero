from replay import SessionRecorder, read_session

def test_replay_roundtrip_preserves_actions_rewards_and_provenance(tmp_path):
    recorder=SessionRecorder(tmp_path,{'seed':17,'driver':'manual'},{'dataset':'test metadata'})
    snapshot={'game':{'score':100,'action':[0,1,0,0],'reward':1}, 'neural':{'time_ms':10,'sample_counts':[2,0]}}
    recorder.write({'kind':'frame','snapshot':snapshot})
    recorder.close()
    result=read_session(recorder.path)
    assert result['frames']==[snapshot]
    assert result['header']['config']['seed']==17
    assert result['header']['provenance']['dataset']=='test metadata'
