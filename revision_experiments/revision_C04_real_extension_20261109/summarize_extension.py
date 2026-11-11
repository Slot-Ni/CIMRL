import json
from collections import defaultdict
import numpy as np
from run_suite import HERE,sha,write

def main():
    groups=defaultdict(list)
    for p in (HERE/'results/extension').glob('*/complete.json'):
        rec=json.loads(p.read_text())
        for rel,h in rec['artifacts'].items():assert sha(p.parent/rel)==h
        row=json.loads((p.parent/rec['attempt']/'results.jsonl').read_text().splitlines()[0])
        groups[(row['dataset'],row['ratio'])].append(row)
    out=[]
    for (ds,ratio),rows in sorted(groups.items()):
        sets=[set(r['selected']) for r in rows]
        jac=[len(a&b)/len(a|b) for i,a in enumerate(sets) for b in sets[i+1:]]
        out.append(dict(dataset=ds,ratio=ratio,n=len(rows),mean=float(np.mean([r['mean'] for r in rows])),sd=float(np.std([r['mean'] for r in rows],ddof=1)) if len(rows)>1 else None,training_minutes=float(np.mean([r['training_seconds'] for r in rows])/60),seed_jaccard=float(np.mean(jac)) if jac else None))
    write(HERE/'summary.json',dict(complete=sum(len(v) for v in groups.values())==35,completed=sum(len(v) for v in groups.values()),groups=out))
    print('Summary saved; prior Bars-Rev ratios remain in verified server archive and are not rerun.')
if __name__=='__main__':main()
