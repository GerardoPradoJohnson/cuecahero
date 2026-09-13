"""JSONL snapshots reproduce the observed run; they are not neural checkpoints."""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

class SessionRecorder:
    def __init__(self, directory, config, provenance):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
        self.path = directory / (self.id + '.jsonl')
        self.stream = self.path.open('x', encoding='utf-8')
        self.write({'kind':'header', 'schema':1, 'id':self.id, 'config':config, 'provenance':provenance,
                    'replay_type':'Recorded observations, actions and activity; not a neural checkpoint'})

    def write(self, value):
        self.stream.write(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n')
        self.stream.flush()

    def close(self):
        if not self.stream.closed:
            self.stream.close()

def read_session(path):
    records = []
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if not line.endswith('\n'):
                break  # A currently recording final line may be incomplete.
            records.append(json.loads(line))
    if not records or records[0].get('schema') != 1:
        raise ValueError('Unsupported replay')
    return {'header':records[0], 'frames':[r['snapshot'] for r in records[1:] if r.get('kind') == 'frame']}
