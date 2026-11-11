"""Run-level uncertainty, seed stability and host-stratified timing."""
import csv
import itertools
from collections import defaultdict
import numpy as np
from scipy.stats import t


def interval(values):
    a = np.asarray(values, dtype=float)
    sd = float(a.std(ddof=1)) if len(a) > 1 else None
    return dict(n=len(a), mean=float(a.mean()), sd=sd,
                ci95_halfwidth=float(t.ppf(.975, len(a)-1)*sd/np.sqrt(len(a))) if sd is not None else None)


def save(path, rows):
    if rows:
        with path.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def summarize(rows, output):
    groups, timing = defaultdict(list), defaultdict(list)
    for row in rows:
        key = (row['dataset'], row['mode'], row['ratio'], row['method'])
        groups[key].append(row)
        timing[key + (row['host'],row.get('execution_backend','unspecified'))].append(row)
    summary = []
    for key, runs in groups.items():
        if len({r['repeat'] for r in runs}) != len(runs):
            raise ValueError('Duplicate repeat in statistical group')
        pairs = [(set(a['selected']), set(b['selected'])) for a,b in itertools.combinations(runs, 2)]
        record = dict(zip(['dataset','mode','ratio','method'], key))
        record.update(interval([r['mean'] for r in runs]))
        record['seed_jaccard'] = float(np.mean([len(a&b)/len(a|b) for a,b in pairs])) if pairs else None
        summary.append(record)
    save(output/'quality_summary.csv', summary)
    times = []
    for key, runs in timing.items():
        record = dict(zip(['dataset','mode','ratio','method','host','execution_backend'], key))
        record['runs'] = len(runs)
        for field in ('preprocessing_seconds','screening_seconds','training_seconds','inference_seconds','selection_seconds','evaluation_seconds'):
            record[field] = float(np.mean([r[field] for r in runs]))
        times.append(record)
    save(output/'timing_by_host.csv', times)
    # Matched independent-repeat contrasts. Same evaluation bank is not event-level coupling.
    contrasts = []
    for dataset in sorted({r['dataset'] for r in rows}):
        for left, right, ratio_left, ratio_right, label in [
            ('dqn','s2v',.2,.2,'C07_dqn_minus_s2v'),
            ('dqn','greedy',.2,.2,'C07_dqn_minus_greedy'),
            *[('dqn','dqn',p,1.0,'C04_screened_minus_full') for p in (.05,.1,.2,.4)]]:
            a = {r['repeat']:r['mean'] for r in rows if r['dataset']==dataset and r['mode']=='product' and r['ratio']==ratio_left and r['method']==left}
            b = {r['repeat']:r['mean'] for r in rows if r['dataset']==dataset and r['mode']=='product' and r['ratio']==ratio_right and r['method']==right}
            common = sorted(a.keys() & b.keys())
            if common:
                contrasts.append(dict(dataset=dataset, comparison=label, ratio=ratio_left, **interval([a[i]-b[i] for i in common])))
    save(output/'matched_quality_differences.csv', contrasts)
