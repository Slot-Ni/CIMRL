"""Report between-training-run uncertainty separately from diffusion variability."""
import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import t

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=Path)
args = parser.parse_args()
groups = defaultdict(list)
for line in (args.directory / 'results.jsonl').read_text().splitlines():
    row = json.loads(line)
    groups[(row['dataset'],row['mode'],row['ratio'],row['method'])].append(row)
records = []
for key, rows in groups.items():
    means = np.array([r['mean'] for r in rows])
    count = len(means)
    sd = float(means.std(ddof=1)) if count>1 else None
    half = float(t.ppf(.975,count-1)*sd/np.sqrt(count)) if count>1 else None
    pairs = [(set(a['selected']),set(b['selected'])) for a,b in itertools.combinations(rows,2)]
    record = dict(zip(['dataset','mode','ratio','method'],key))
    record.update(repeats=count,mean=float(means.mean()),between_run_sd=sd,
                  ci95_halfwidth=half,mean_pairwise_jaccard=np.mean([len(a&b)/len(a|b) for a,b in pairs]) if pairs else None)
    for field in ['preprocessing_seconds','screening_seconds','training_seconds',
                  'inference_seconds','selection_seconds','evaluation_seconds']:
        record[field]=float(np.mean([r[field] for r in rows]))
    records.append(record)
if records:
    with (args.directory/'summary.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
