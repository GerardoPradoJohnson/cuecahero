"""HTTP boundary + manual worker lifecycle; no synthetic neural runtime."""
import json
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import pytest
from core.paths import ROOT
from experiments.runtime import Experiment
from scripts.serve import make_handler
from replay import read_session

@pytest.fixture
def application(tmp_path):
    config=json.loads((ROOT/'experiments/cueca_hero.json').read_text())
    experiment=Experiment(config, {'total_neurons':0,'indices':[]}, sessions_dir=tmp_path)
    server=ThreadingHTTPServer(('127.0.0.1',0),BaseHTTPRequestHandler)
    server.RequestHandlerClass=make_handler(experiment,{},server.server_port)
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    yield experiment, f'http://127.0.0.1:{server.server_port}', tmp_path
    server.shutdown()
    server.server_close()
    experiment.close()
    thread.join(timeout=2)


def post(url, command, origin=None):
    headers={'Content-Type':'application/json'}
    if origin:
        headers['Origin']=origin
    return urlopen(Request(url+'/api/control',data=json.dumps(command).encode(),headers=headers),timeout=3)


def wait_for(predicate):
    deadline=time.monotonic()+3
    while time.monotonic()<deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('Worker state did not converge')


def test_http_control_lifecycle_records_inputs_and_pause(application):
    experiment,url,directory=application
    assert post(url,{'type':'start'}).status==202
    wait_for(lambda:experiment.snapshot['mode']=='running')
    post(url,{'type':'keys','lanes':[0]})
    wait_for(lambda:experiment.snapshot['game']['wrong']==1)
    post(url,{'type':'pause'})
    wait_for(lambda:experiment.snapshot['mode']=='paused')
    paused_time=experiment.snapshot['game']['time']
    time.sleep(.1)
    assert experiment.snapshot['game']['time']==paused_time
    session=experiment.last_session
    frames=read_session(directory/(session+'.jsonl'))['frames']
    assert any(f['game']['action']==[1,0,0,0] for f in frames)
    assert all(f['neural']['time_ms']==0 for f in frames)
    post(url,{'type':'reset'})
    wait_for(lambda:experiment.snapshot['game']['time']==0)
    assert experiment.snapshot['game']['wrong']==0
    assert experiment.snapshot['telemetry']['total_spikes']==0


def test_invalid_commands_and_foreign_origin_leave_session_unchanged(application):
    experiment,url,_=application
    with pytest.raises(HTTPError) as bad:
        post(url,{'type':'keys','lanes':[-1]})
    assert bad.value.code==400
    with pytest.raises(HTTPError) as foreign:
        post(url,{'type':'start'},'https://example.invalid')
    assert foreign.value.code==403
    with pytest.raises(HTTPError) as missing:
        urlopen(url+'/api/replay?id=../../README')
    assert missing.value.code==404
    assert experiment.snapshot['mode']=='paused'
