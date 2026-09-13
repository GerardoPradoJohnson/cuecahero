"""Measured soma coordinates only. Sampling affects rendering, never simulation."""
import hashlib
import json
import numpy as np
from core.paths import DATA, ROOT

def load_geometry(sample_size=3200, max_edges=4500):
    import pyarrow.feather as feather
    source = DATA / 'connectome_data/malecns_v1/annotations.feather'
    expected = json.loads((ROOT / 'config/malecns.lock.json').read_text())['annotations.feather']
    if source.stat().st_size != expected['bytes'] or hashlib.sha256(source.read_bytes()).hexdigest() != expected['sha256']:
        raise ValueError('Soma annotation checksum mismatch')
    nodes = feather.read_table(DATA/'connectome_data/malecns_v1/normalized/neurons.feather', columns=['source_id','superclass','cell_type','neurotransmitter']).to_pandas()
    annotations = feather.read_table(source, columns=['bodyId','somaLocation']).to_pandas().set_index('bodyId').reindex(nodes.source_id)
    valid = [(i, p) for i, p in enumerate(annotations.somaLocation) if isinstance(p, (list, np.ndarray)) and len(p) == 3 and np.isfinite(p).all()]
    valid_dict = dict(valid)
    valid_idx = [i for i, _ in valid]

    nt = nodes['neurotransmitter'].values
    sc = nodes['superclass'].values
    ct = nodes['cell_type'].fillna('').astype(str)

    is_dopa = (nt == 'dopamine')
    is_motor = np.isin(sc, ['descending_neuron', 'vnc_motor', 'cb_motor']) & ~is_dopa
    is_auditory = (ct.str.startswith('AMMC') | ct.str.startswith('WED')).values & ~is_dopa & ~is_motor
    is_sensory = (np.isin(sc, ['visual_projection', 'ol_sensory', 'cb_sensory', 'vnc_sensory']) | (nt == 'histamine')) & ~is_dopa & ~is_motor & ~is_auditory

    dopa_valid = [i for i in valid_idx if is_dopa[i]]
    motor_valid = [i for i in valid_idx if is_motor[i]]
    auditory_valid = [i for i in valid_idx if is_auditory[i]]
    sensory_valid = [i for i in valid_idx if is_sensory[i]]
    intrinsic_valid = [i for i in valid_idx if not (is_dopa[i] or is_motor[i] or is_auditory[i] or is_sensory[i])]

    target_dopa = min(len(dopa_valid), max(15, int(sample_size * 0.06)))
    target_motor = min(len(motor_valid), max(25, int(sample_size * 0.12)))
    target_auditory = min(len(auditory_valid), max(25, int(sample_size * 0.12)))
    target_sensory = min(len(sensory_valid), max(35, int(sample_size * 0.20)))
    target_intrinsic = max(20, sample_size - (target_dopa + target_motor + target_auditory + target_sensory))

    sel_dopa = [dopa_valid[j] for j in np.linspace(0, len(dopa_valid)-1, target_dopa, dtype=int)]
    sel_motor = [motor_valid[j] for j in np.linspace(0, len(motor_valid)-1, target_motor, dtype=int)]
    sel_auditory = [auditory_valid[j] for j in np.linspace(0, len(auditory_valid)-1, target_auditory, dtype=int)]
    sel_sensory = [sensory_valid[j] for j in np.linspace(0, len(sensory_valid)-1, target_sensory, dtype=int)]
    sel_intrinsic = [intrinsic_valid[j] for j in np.linspace(0, len(intrinsic_valid)-1, target_intrinsic, dtype=int)]

    selected_indices = np.array(sorted(sel_dopa + sel_motor + sel_auditory + sel_sensory + sel_intrinsic), dtype=np.int32)

    points = np.array([valid_dict[i] for i in selected_indices], dtype=float)
    # Uniform scaling preserves relative spatial distances; no inferred positions.
    low, high = points.min(axis=0), points.max(axis=0)
    normalized = (points - (low + high)/2) / max(high - low) * 2

    roles = []
    auditory_local_indices = []
    for local_i, idx in enumerate(selected_indices):
        if is_dopa[idx]:
            roles.append('dopamine')
        elif is_motor[idx]:
            roles.append('motor')
        elif is_auditory[idx]:
            roles.append('auditory')
            auditory_local_indices.append(local_i)
        elif is_sensory[idx]:
            roles.append('stimulus')
        else:
            roles.append('intrinsic')

    # Extract real biological synapses from GRAPH connecting the selected somas
    from core.paths import GRAPH
    g = np.load(GRAPH)
    ptr, post, weight = g['ptr'], g['post'], g['weight']
    sample_set = set(selected_indices)
    idx_map = {idx: i for i, idx in enumerate(selected_indices)}

    raw_edges = []
    for pre_local, pre in enumerate(selected_indices):
        p_start, p_end = ptr[pre], ptr[pre+1]
        targets = post[p_start:p_end]
        w = weight[p_start:p_end]
        for target, wt in zip(targets, w):
            if target in sample_set:
                raw_edges.append((pre_local, idx_map[target], float(wt)))

    raw_edges.sort(key=lambda e: abs(e[2]), reverse=True)
    selected_edges = [[e[0], e[1], round(e[2], 2)] for e in raw_edges[:max_edges]]

    return {
        'indices': selected_indices.tolist(),
        'ids': [str(x) for x in nodes.source_id.iloc[selected_indices]],
        'classes': nodes.superclass.iloc[selected_indices].fillna('unknown').tolist(),
        'roles': roles,
        'role_counts': {
            'stimulus': len(sel_sensory),
            'auditory': len(sel_auditory),
            'motor': len(sel_motor),
            'dopamine': len(sel_dopa),
            'intrinsic': len(sel_intrinsic)
        },
        'auditory_indices': auditory_local_indices,
        'points': np.round(normalized, 6).tolist(),
        'edges': selected_edges,
        'sample_size': len(selected_indices),
        'edge_count': len(selected_edges),
        'total_neurons': len(nodes),
        'neurons_with_soma': len(valid),
        'source': 'MaleCNS v1.0 · somaLocation & synapses',
        'coordinate_units': 'source coordinates, uniformly normalized for display',
        'anatomy': 'Measured soma locations and biological synapses from MaleCNS v1.0'
    }
