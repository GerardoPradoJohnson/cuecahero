import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('FLYLAB_DATA_DIR', ROOT / 'data')).expanduser().resolve()
GRAPH = DATA / 'outputs/doom/malecns_v1/graph.npz'
