"""Select hysteresis thresholds on the recorded training trials only.

Use with calibrate_perception.py's 20-trial population calibration protocol.
This never consumes validation results or alters connectome weights.
"""
import argparse,json,numpy as np
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--calibration',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():parser.error('Output exists; choose a new directory')
p=args.calibration;m=json.loads((p/'model.json').read_text());d=np.load(p/'training.npz');X=d['features'];scores=np.clip(((X-m['mean'])/m['scale'])@np.array(m['weights'])+m['bias'],0,1).reshape(20,108,4)
r=[]
for threshold in np.arange(.30,.76,.025):
 for release in [.1,.15,.2,.25,.3]:
  hit=wrong=miss=0;errors=[]
  for trial,s in enumerate(scores):
   lane=trial%5-1;onset=[1.1,1.57,2.31,2.79][trial//5];armed=np.ones(4,bool);last=np.full(4,-1000);got=False
   for tick,row in enumerate(s):
    armed|=row<release;fire=(row>=threshold)&armed&(tick-last>=3);armed[fire]=False;last[fire]=tick
    for c in np.flatnonzero(fire):
     err=(tick+1)/30-onset
     if c==lane and abs(err)<=.15 and not got:hit+=1;got=True;errors.append(abs(err))
     else:wrong+=1
   if lane>=0 and not got:miss+=1
  r.append((hit-wrong*.6,hit,-wrong,-np.mean(errors) if errors else -100,threshold,release))
r.sort(reverse=True)
print(r[:15]);best=r[0];m['threshold']=float(best[-2]);m['release']=float(best[-1]);m['threshold_selection']={'method':'grid on training trials only; hit - 0.6 * false press','hits':best[1],'false_presses':-best[2]}
out=args.output;out.mkdir(parents=True);(out/'model.json').write_text(json.dumps(m,indent=2)+'\n')
