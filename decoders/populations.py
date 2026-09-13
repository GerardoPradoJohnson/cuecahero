"""Readout pools traced through actual R1–R6→lamina graph edges."""
import numpy as np

def lamina_populations(state, encoder, cell_types, columns=4, rows=8):
    types=np.asarray(cell_types)
    allowed=np.isin(types,['L1','L2','L3','L5'])
    mass=np.zeros(state.n);xy=np.zeros((state.n,2))
    for source,uv in zip(encoder.retina,encoder.uv):
        start,end=state.ptr[source:source+2]
        targets=state.post[start:end];weights=np.abs(state.weight[start:end]).astype(float)
        keep=allowed[targets]
        targets,weights=targets[keep],weights[keep]
        np.add.at(mass,targets,weights)
        np.add.at(xy,targets,weights[:,None]*uv)
    mapped=np.flatnonzero(mass>0)
    xy[mapped]/=mass[mapped,None]
    pools=[];labels=[]
    for kind in ['L1','L2','L3','L5']:
        for y in range(rows):
            for x in range(columns):
                indices=mapped[(types[mapped]==kind)&(np.minimum((xy[mapped,0]*columns).astype(int),columns-1)==x)&(np.minimum((xy[mapped,1]*rows).astype(int),rows-1)==y)]
                if len(indices)>=3:
                    pools.append(indices.tolist());labels.append({'type':kind,'x_bin':x,'y_bin':y,'neurons':len(indices)})
    if not pools:raise ValueError('No graph-supported lamina populations')
    return pools,labels
