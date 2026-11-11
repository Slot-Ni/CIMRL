"""Portable deterministic task plans, disjoint shards, task-level resume and audited merge."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
import threading
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
S2V_SLOT = threading.Semaphore(1)
EXPERIMENTS = ['C04_candidate_pool_sensitivity', 'C05_component_ablation', 'C07_fair_budget']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.writing')
    with temporary.open('w', encoding='utf-8') as file:
        file.write(json.dumps(value, indent=2, ensure_ascii=False))
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def make_plan(path):
    jobs = {}
    for experiment in EXPERIMENTS:
        config = json.loads((HERE / (experiment + '.json')).read_text(encoding='utf-8-sig'))
        for dataset in config['datasets']:
            for repeat in range(config['repeats']):
                for ratio in config['ratios']:
                    for mode in config['modes']:
                        for method in config['methods']:
                            cfg = {k: v for k, v in config.items() if k not in
                                   ('experiment_id', 'review_comment_ids', 'datasets', 'ratios', 'modes', 'methods', 'data_root', 'cache_root', 's2v_embedding_dim', 's2v_iterations')}
                            cfg.update(datasets=[dataset], repeat_ids=[repeat], ratios=[ratio], modes=[mode], methods=[method])
                            if method == 's2v':
                                cfg.update(s2v_embedding_dim=32, s2v_iterations=32)
                            task_id = identity(cfg)[:20]
                            if task_id not in jobs:
                                jobs[task_id] = dict(task_id=task_id, config=cfg, experiments=[])
                            jobs[task_id]['experiments'].append(experiment)
    tasks = sorted(jobs.values(), key=lambda j: j['task_id'])
    plan = dict(version=1, tasks=tasks, code_sha256={p.name: sha(p) for p in sorted(HERE.glob('*.py'))})
    plan['plan_id'] = identity(plan)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != plan:
            raise ValueError('Existing plan differs; choose a new plan path')
    else:
        write(path, plan)
    print(f"{len(tasks)} unique tasks; plan {plan['plan_id'][:16]}", flush=True)
    return plan


def load_plan(path):
    plan = json.loads(path.read_text(encoding='utf-8'))
    if identity({k: v for k, v in plan.items() if k != 'plan_id'}) != plan['plan_id']:
        raise ValueError('Plan checksum mismatch')
    for name, expected in plan['code_sha256'].items():
        if sha(HERE / name) != expected:
            raise ValueError(f'Code differs from plan: {name}; use original code or create a new plan')
    return plan


def preflight(data, cache, plan):
    import numpy as np
    names = sorted({j['config']['datasets'][0] for j in plan['tasks']})
    sources = {}
    for name in names:
        matrix_path = next((p for p in (data / (name + '.npy'), data / 'HyperGraphMatrix' / (name + '.npy')) if p.exists()), None)
        if matrix_path is None:
            raise FileNotFoundError(name)
        matrix = np.load(matrix_path, mmap_mode='r', allow_pickle=False)
        if matrix.ndim != 2 or not np.isin(matrix, [0, 1]).all():
            raise ValueError(f'Invalid matrix: {matrix_path}')
        orders = max(j['config']['orders'] for j in plan['tasks'] if j['config']['datasets'][0] == name)
        files = [matrix_path]
        for order in range(1, orders + 1):
            filename = f'{name}_distance_{order}.npy'
            path = next((p for p in (cache / filename, cache / 'Hyper_Distance' / filename) if p.exists()), None)
            if path is None:
                raise FileNotFoundError(f'{filename}; recomputation disabled')
            a = np.load(path, mmap_mode='r', allow_pickle=False)
            if a.shape != (len(matrix), len(matrix)):
                raise ValueError(f'Cache shape mismatch: {path}')
            files.append(path)
        sources[name] = {p.name: sha(p) for p in files}
        print(f'{name}: matrix {matrix.shape}, {orders} cache orders present; SHA256 recorded', flush=True)
    return sources


def run(args):
    global S2V_SLOT
    if args.s2v_workers < 1:
        raise ValueError('s2v-workers must be positive')
    S2V_SLOT = threading.Semaphore(args.s2v_workers)
    if args.workers < 1:
        raise ValueError('workers must be positive')
    if not 0 <= args.shard_index < args.shards:
        raise ValueError('Require 0 <= shard-index < shards')
    plan = load_plan(args.plan)
    sources = preflight(args.data_root.resolve(), args.cache_root.resolve(), plan)
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [j for i, j in enumerate(plan['tasks']) if i % args.shards == args.shard_index]
    if args.limit:
        jobs = jobs[:args.limit]
    if args.task_file:
        selected_ids = set(json.loads(args.task_file.read_text(encoding='utf-8')))
        unknown = selected_ids - {j['task_id'] for j in plan['tasks']}
        if unknown:
            raise ValueError(f'Unknown task IDs: {unknown}')
        jobs = [j for j in jobs if j['task_id'] in selected_ids]
    jobs.sort(key=lambda j: (0 if list((args.output/j['task_id']).glob('attempt_*/*/*_training_latest.pt')) and not (args.output/j['task_id']/'complete.json').exists() else {'ranking':1,'dqn':2,'greedy':2,'s2v':3}[j['config']['methods'][0]], j['task_id']))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_job, job, args, plan, sources): job['task_id'] for job in jobs}
        try:
            for future in tqdm(as_completed(futures), total=len(futures), desc=f'Worker {args.shard_index}', unit='task'):
                future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def run_job(job, args, plan, sources):
    directory = args.output / job['task_id']
    directory.mkdir(exist_ok=True)
    done = directory / 'complete.json'
    if done.exists():
        record = json.loads(done.read_text(encoding='utf-8'))
        if record['plan_id'] != plan['plan_id'] or record['sources'] != sources:
            raise ValueError('Completed task has different plan/data')
        verify_artifacts(directory, record)
        print('SKIP', job['task_id'], flush=True)
        return
    lock = directory / 'RUNNING.lock'
    # Atomic across processes using the same output filesystem.
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    os.write(fd, json.dumps(dict(host=platform.node(), pid=os.getpid(), started=time.time())).encode())
    os.close(fd)
    try:
        provenance = directory / 'run_identity.json'
        expected_identity = dict(plan_id=plan['plan_id'], sources=sources)
        if provenance.exists() and json.loads(provenance.read_text(encoding='utf-8')) != expected_identity:
            raise ValueError('Incomplete task belongs to different code/plan/data; use a new output directory')
        write(provenance, expected_identity)
        attempt = directory / ('attempt_' + uuid.uuid4().hex[:12])
        cfg = dict(job['config'], data_root=str(args.data_root.resolve()), cache_root=str(args.cache_root.resolve()), progress_every=args.progress_every, s2v_microbatch=8)
        previous_attempts = sorted([d for d in directory.glob('attempt_*') if d.is_dir() and list(d.rglob('*_training_latest.pt'))], key=lambda d: d.stat().st_mtime)
        if previous_attempts:
            cfg['resume_directory'] = str(previous_attempts[-1].resolve())
            if (previous_attempts[-1] / 'resume_origin.json').exists():
                cfg['resume_source_backend'] = json.loads((previous_attempts[-1] / 'resume_origin.json').read_text(encoding='utf-8'))['backend']
        config_path = directory / ('config_' + attempt.name + '.json')
        write(config_path, cfg)
        print(f"TASK {job['task_id']} {cfg['datasets']} {cfg['methods']} repeat={cfg['repeat_ids']}", flush=True)
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
        with (S2V_SLOT if cfg['methods'] == ['s2v'] else nullcontext()):
            with (directory / 'console.log').open('a', encoding='utf-8') as log:
                subprocess.run([sys.executable, '-u', str(HERE / 'shared_experiment_core.py'), '--config', str(config_path), '--output', str(attempt)], check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
        artifacts = {str(p.relative_to(directory)): sha(p) for p in attempt.rglob('*') if p.is_file()}
        record = dict(task=job, plan_id=plan['plan_id'], sources=sources, attempt=attempt.name, artifacts=artifacts)
        temporary = directory / 'complete.tmp'
        write(temporary, record)
        temporary.replace(done)
    finally:
        lock.unlink(missing_ok=True)


def verify_artifacts(directory, record):
    for name, expected in record['artifacts'].items():
        if sha(directory / name) != expected:
            raise ValueError(f'Artifact changed: {directory / name}')


def merge(args):
    plan = load_plan(args.plan)
    expected = {j['task_id']: j for j in plan['tasks']}
    found, source_reference = {}, None
    for root in args.inputs:
        for done in root.glob('*/complete.json'):
            record = json.loads(done.read_text(encoding='utf-8'))
            task_id = record['task']['task_id']
            if record['plan_id'] != plan['plan_id'] or task_id not in expected or record['task'] != expected[task_id]:
                raise ValueError(f'Foreign/mismatched task: {done}')
            verify_artifacts(done.parent, record)
            if task_id in found:
                raise ValueError(f'Duplicate task {task_id}; remove duplicate input, never silently average')
            if source_reference is not None and source_reference != record['sources']:
                raise ValueError('Input matrix/cache hashes differ across workers')
            source_reference = record['sources']
            found[task_id] = (done.parent, record)
    missing = sorted(set(expected) - set(found))
    if missing and not args.allow_partial:
        raise ValueError(f'{len(missing)} missing tasks; --allow-partial is for progress inspection only')
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for task_id, (directory, record) in sorted(found.items()):
        attempt = directory / record['attempt']
        manifest = json.loads((attempt / 'manifest.json').read_text(encoding='utf-8'))
        for line in (attempt / 'results.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            row.update(execution_backend=row.get('fast_backend','reference'), task_id=task_id, experiments=record['task']['experiments'], host=manifest['host'], result_directory=str(attempt.resolve()), plan_id=plan['plan_id'])
            rows.append(row)
    (args.output / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    write(args.output / 'merge_manifest.json', dict(plan_id=plan['plan_id'], expected=len(expected), completed=len(found), missing=missing, complete=not missing, sources=source_reference))
    from summarize_suite import summarize
    summarize(rows, args.output)
    print(f'Merged {len(found)}/{len(expected)} tasks. Raw artifacts remain in worker output folders.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('plan'); s.add_argument('--plan', type=Path, default=HERE/'plan.json')
    for cmd in ('check', 'run'):
        s = sub.add_parser(cmd)
        s.add_argument('--plan', type=Path, default=HERE/'plan.json')
        s.add_argument('--data-root', type=Path, default=HERE/'../../HyperGraphMatrix')
        s.add_argument('--cache-root', type=Path, default=HERE/'../../Hyper_Distance')
        if cmd == 'run':
            s.add_argument('--output', type=Path, required=True)
            s.add_argument('--shards', type=int, default=1)
            s.add_argument('--shard-index', type=int, default=0)
            s.add_argument('--limit', type=int)
            s.add_argument('--workers', type=int, default=2)
            s.add_argument('--s2v-workers', type=int, default=1)
            s.add_argument('--task-file', type=Path)
            s.add_argument('--progress-every', type=int, default=10)
    s = sub.add_parser('merge')
    s.add_argument('--plan', type=Path, default=HERE/'plan.json')
    s.add_argument('--inputs', type=Path, nargs='+', required=True)
    s.add_argument('--output', type=Path, required=True)
    s.add_argument('--allow-partial', action='store_true')
    args = p.parse_args()
    if args.command == 'plan': make_plan(args.plan)
    elif args.command == 'check':
        plan = load_plan(args.plan)
        preflight(args.data_root.resolve(), args.cache_root.resolve(), plan)
    elif args.command == 'run': run(args)
    else: merge(args)


if __name__ == '__main__':
    main()
