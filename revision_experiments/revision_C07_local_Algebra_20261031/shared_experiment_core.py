"""Isolated revision experiments; row indices are used throughout."""
import argparse
import hashlib
import json
import os
import platform
import time
from collections import deque, OrderedDict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm


def append_durable(path, record):
    with Path(path).open('a', encoding='utf-8') as file:
        file.write(json.dumps(record) + '\n')
        file.flush()
        os.fsync(file.fileno())


def save_checkpoint(path, state):
    import torch
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as file:
        torch.save(state, file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def structure(matrix):
    return ([np.flatnonzero(row) for row in matrix],
            [np.flatnonzero(col) for col in matrix.T])


def separation(matrix, rivals, orders, alpha):
    """Exact multi-source edge BFS per rival/order; never reuse legacy caches.

    Computes only the distances actually needed for xi, retaining infinity.
    Shared-edge distance=1 and self-distance=1, including isolated nodes.
    """
    incident, members = structure(matrix)
    sparse = csr_matrix(matrix.astype(np.int64))
    overlaps = (sparse.T @ sparse).tocsr()
    xi = np.zeros(len(matrix))
    all_distances = []
    for order in range(1, orders + 1):
        adjacency = []
        for edge in range(matrix.shape[1]):
            start, stop = overlaps.indptr[edge:edge + 2]
            ids, counts = overlaps.indices[start:stop], overlaps.data[start:stop]
            adjacency.append(ids[(counts >= order) & (ids != edge)])
        distances = np.full((len(rivals), len(matrix)), np.inf)
        for k, rival in enumerate(rivals):
            edge_dist = np.full(matrix.shape[1], np.inf)
            edge_dist[incident[rival]] = 0
            queue = deque(incident[rival].tolist())
            while queue:
                edge = queue.popleft()
                for other in adjacency[edge]:
                    if np.isinf(edge_dist[other]):
                        edge_dist[other] = edge_dist[edge] + 1
                        queue.append(int(other))
            for node, edges in enumerate(incident):
                if len(edges):
                    distances[k, node] = 1 + edge_dist[edges].min()
            distances[k, rival] = 1
        xi += alpha ** (-order) * np.where(np.isfinite(distances), distances, 0).mean(axis=0)
        all_distances.append(distances)
    xi[rivals] = 0
    phi = np.array([np.mean([len(members[e]) for e in edges]) if len(edges) else 0
                    for edges in incident])
    return xi, phi, np.array(all_distances)


def cached_separation(matrix, rivals, orders, alpha, name, cache_root):
    """Read existing cache only; preserve legacy matrix[node, rival] orientation."""
    root = Path(cache_root)
    xi = np.zeros(len(matrix))
    distances, sources = [], []
    for order in range(1, orders + 1):
        filename = f'{name}_distance_{order}.npy'
        path = root / filename
        if not path.exists():
            path = root / 'Hyper_Distance' / filename
        if not path.exists():
            raise FileNotFoundError(f'Missing existing cache: {filename}; recomputation disabled')
        cached = np.load(path, mmap_mode='r', allow_pickle=False)
        if cached.shape != (len(matrix), len(matrix)):
            raise ValueError(f'Cache shape mismatch: {path}')
        selected = np.asarray(cached[:, rivals], dtype=float).T.copy()
        if np.isnan(selected).any() or (selected < 0).any():
            raise ValueError(f'Invalid cached distance values: {path}')
        xi += alpha ** (-order) * np.where(np.isfinite(selected), selected, 0).mean(axis=0)
        distances.append(selected)
        sources.append({'order': order, 'path': str(path.resolve()),
                        'size_bytes': path.stat().st_size, 'sha256': digest(path)})
    xi[rivals] = 0
    degrees = matrix.sum(axis=1)
    phi = np.divide(matrix @ matrix.sum(axis=0), degrees,
                    out=np.zeros(len(matrix), dtype=float), where=degrees > 0)
    return xi, phi, np.array(distances), sources


class Diffusion:
    """Equal beta, target-priority ties, irreversible synchronous activation.

    Local RNG only. Same seed is NOT claimed to be event-level coupling.
    """
    def __init__(self, matrix, rivals, beta, horizon):
        self.matrix = matrix
        self.incident, self.members = structure(matrix)
        self.rivals, self.beta, self.horizon = rivals, beta, horizon

    def trial(self, selected, seed):
        rng = np.random.default_rng(seed)
        n = len(self.incident)
        state = np.zeros(n, dtype=np.int8)
        state[self.rivals], state[selected] = -1, 1
        prepared = getattr(self, '_choice_bank', {}).get(int(seed))
        if prepared is None:
            choices = [rng.choice(edges, self.horizon) if len(edges)
                       else np.full(self.horizon, -1) for edges in self.incident]
        else:
            choices, after_choices = prepared
            rng.bit_generator.state = after_choices
        for t in range(self.horizon):
            old = state.copy()
            for kind in (1, -1):
                edges = [choices[u][t] for u in np.flatnonzero(old == kind)
                         if choices[u][t] >= 0]
                if not edges:
                    continue
                neighbors = np.concatenate([self.members[e] for e in edges])
                susceptible = neighbors[state[neighbors] == 0]
                activated = susceptible[rng.random(len(susceptible)) < self.beta]
                state[activated] = kind
        return int(np.sum(state == 1))

    def samples(self, selected, seeds):
        # Training uses a fixed bank. Cache exact deterministic evaluations, never estimates
        # from a different seed bank. The original random draw order is preserved.
        seed_key = tuple(int(s) for s in seeds)
        key = (tuple(sorted(int(v) for v in selected)), seed_key)
        if not hasattr(self, '_reward_cache'):
            self._reward_cache = OrderedDict()
            self._choice_bank = {}
            self.cache_hits = self.cache_misses = 0
        if key in self._reward_cache:
            self.cache_hits += 1
            self._reward_cache.move_to_end(key)
            return self._reward_cache[key].copy()
        self.cache_misses += 1
        if len(seed_key) <= 64:
            if set(self._choice_bank) != set(seed_key):
                self._choice_bank = {}
                for seed in seed_key:
                    rng = np.random.default_rng(seed)
                    choices = [rng.choice(edges, self.horizon) if len(edges)
                               else np.full(self.horizon, -1) for edges in self.incident]
                    self._choice_bank[seed] = (choices, rng.bit_generator.state)
        result = np.array([self.trial(selected, int(seed)) for seed in seeds])
        self._reward_cache[key] = result
        if len(self._reward_cache) > 20000:
            self._reward_cache.popitem(last=False)
        return result.copy()


def pool_for(xi, phi, rivals, ratio, budget, mode):
    score = {'product': xi * phi, 'xi': xi, 'phi': phi}[mode]
    eligible = np.setdiff1d(np.arange(len(xi)), rivals)
    # Preserve historical ratio denominator n; enforce budget feasibility.
    size = min(len(eligible), max(budget, int(ratio * len(xi))))
    return eligible[np.lexsort((eligible, -score[eligible]))][:size]


def learn(pool, features, budget, simulator, seeds, cfg, seed):
    import torch
    from torch import nn
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    def network():
        return nn.Sequential(nn.Linear(2 * budget, 256), nn.LayerNorm(256), nn.ReLU(),
                             nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(),
                             nn.Linear(128, len(pool)))
    if cfg.get('learner') == 's2v':
        from s2v_adapter import make_network
        network = lambda: make_network(simulator.matrix, simulator.rivals, pool, cfg)
    model, target = network(), network()
    target.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    memory = deque(maxlen=cfg['replay_capacity'])
    steps = 0
    first_episode, elapsed = 0, 0.0
    checkpoint_path = Path(cfg['checkpoint_path']) if cfg.get('checkpoint_path') else None
    resume_path = Path(cfg['resume_checkpoint']) if cfg.get('resume_checkpoint') else checkpoint_path
    if resume_path is not None and resume_path.exists():
        saved = torch.load(resume_path, map_location='cpu', weights_only=False)
        if saved['seed'] != seed or saved['pool'] != pool.tolist() or saved['episodes'] != cfg['episodes']:
            raise ValueError('Training checkpoint identity mismatch')
        model.load_state_dict(saved['model'])
        target.load_state_dict(saved['target'])
        optimizer.load_state_dict(saved['optimizer'])
        memory = deque(saved['replay'], maxlen=cfg['replay_capacity'])
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'])
        first_episode, steps, elapsed = saved['next_episode'], saved['steps'], saved['training_seconds']
        print(f'Resume training at episode {first_episode}/{cfg["episodes"]}', flush=True)
    def encode(actions):
        if cfg.get('learner') == 's2v':
            result = np.zeros(len(features), dtype=np.float32)
            result[pool[actions]] = 1
            return result
        result = np.zeros(2 * budget, dtype=np.float32)
        result[:2 * len(actions)] = features[pool[actions]].ravel()
        return result
    for episode in tqdm(range(first_episode, cfg['episodes']), total=cfg['episodes'], initial=first_episode, desc=f"Train {cfg.get('learner', 'dqn')}", unit='episode'):
        actions, previous = [], 0.0
        epsilon = .5 + (.01 - .5) * episode / max(1, cfg.get('epsilon_schedule_episodes', cfg['episodes']) - 1)
        for step in range(budget):
            state = encode(actions)
            legal = np.ones(len(pool), dtype=bool)
            legal[actions] = False
            if rng.random() < epsilon:
                action = int(rng.choice(np.flatnonzero(legal)))
            else:
                with torch.no_grad():
                    values = model(torch.from_numpy(state)).numpy()
                action = int(np.argmax(np.where(legal, values, -np.inf)))
            actions.append(action)
            value = float(simulator.samples(pool[actions].tolist(), seeds).mean())
            next_legal = legal.copy()
            next_legal[action] = False
            done = len(actions) == budget
            memory.append((state, action, value - previous, encode(actions), next_legal, done))
            previous = value
            if len(memory) >= cfg['batch_size']:
                batch = [memory[i] for i in rng.choice(len(memory), cfg['batch_size'], replace=False)]
                states, acts, rewards, following, masks, dones = map(np.array, zip(*batch))
                # Effective replay batch stays 64. Accumulate smaller S2V microbatches
                # to bound activation memory; there is one Adam update per full batch.
                chunk_size = cfg.get('s2v_microbatch', 8) if cfg.get('learner') == 's2v' else len(batch)
                optimizer.zero_grad()
                for begin in range(0, len(batch), chunk_size):
                    end = min(len(batch), begin + chunk_size)
                    with torch.no_grad():
                        q_next = target(torch.tensor(following[begin:end])).masked_fill(~torch.tensor(masks[begin:end]), -torch.inf)
                        maximum = q_next.max(dim=1).values
                        maximum[torch.tensor(dones[begin:end])] = 0
                        expected = torch.tensor(rewards[begin:end], dtype=torch.float32) + cfg['gamma'] * maximum
                    predicted = model(torch.tensor(states[begin:end])).gather(1, torch.tensor(acts[begin:end])[:, None]).squeeze(1)
                    loss = nn.functional.mse_loss(predicted, expected, reduction='sum') / len(batch)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite DQN loss')
                    loss.backward()
                optimizer.step()
            steps += 1
            if steps % cfg['target_update_steps'] == 0:
                target.load_state_dict(model.state_dict())
        if cfg.get('training_log'):
            append_durable(cfg['training_log'], dict(episode=episode+1, influence=previous, epsilon=epsilon, steps=steps))
        if checkpoint_path is not None:
            save_checkpoint(checkpoint_path, dict(model=model.state_dict(), target=target.state_dict(),
                optimizer=optimizer.state_dict(), replay=list(memory), numpy_rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(), next_episode=episode+1, steps=steps, seed=seed,
                pool=pool.tolist(), episodes=cfg['episodes'], training_seconds=elapsed+time.perf_counter()-start))
    training_seconds = elapsed + time.perf_counter() - start
    start = time.perf_counter()
    actions = []
    with torch.no_grad():
        for _ in range(budget):
            values = model(torch.from_numpy(encode(actions))).numpy()
            values[actions] = -np.inf
            actions.append(int(np.argmax(values)))
    return pool[actions].tolist(), training_seconds, time.perf_counter() - start, model.state_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    for key in ('data_root', 'cache_root'):
        location = Path(cfg[key])
        if not location.is_absolute():
            location = args.config.resolve().parent / location
        cfg[key] = str(location.resolve())
    if not 0 < cfg['alpha'] or cfg['orders'] < 1 or cfg['budget'] < 1:
        raise ValueError('Invalid structural parameters')
    if not 0 <= cfg['beta'] <= 1 or not 0 <= cfg['gamma'] <= 1:
        raise ValueError('Invalid beta/gamma')
    if any(not 0 < ratio <= 1 for ratio in cfg['ratios']):
        raise ValueError('Ratio must be in (0,1]')
    if min(cfg['train_simulations'], cfg['eval_simulations'], cfg['repeats'], cfg['episodes']) < 1:
        raise ValueError('Simulation and training counts must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(cfg['data_root'])
    manifest = {'config': cfg, 'runner_sha256': digest(__file__), 'python': platform.python_version(),
                'numpy': np.__version__, 'host': platform.node(), 'platform': platform.platform(),
                'processor': platform.processor(), 'code_sha256': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}, 'semantics': 'equal-beta,target-priority,synchronous',
                'historical_results_verified': False, 'datasets': {}}
    for name in cfg['datasets']:
        path = root / (name + '.npy')
        if not path.exists():
            path = root / 'HyperGraphMatrix' / (name + '.npy')
        matrix = np.load(path, allow_pickle=False)
        if matrix.ndim != 2 or not np.isin(matrix, [0, 1]).all():
            raise ValueError('Expected a binary incidence matrix')
        n = len(matrix)
        rival_count = cfg.get('rival_count', max(1, int(.01 * n)))
        rivals = np.lexsort((np.arange(n), -matrix.sum(axis=1)))[:rival_count]
        if not 1 <= rival_count <= n - cfg['budget']:
            raise ValueError('Infeasible competing/target budgets')
        directory = args.output / name
        directory.mkdir()
        start = time.perf_counter()
        xi, phi, distances, cache_sources = cached_separation(
            matrix, rivals, cfg['orders'], cfg['alpha'], name, cfg['cache_root'])
        preprocessing = time.perf_counter() - start
        np.savez_compressed(directory / 'features.npz', xi=xi, phi=phi, rivals=rivals, distances=distances)
        manifest['datasets'][name] = {'matrix': str(path), 'sha256': digest(path),
                                     'shape': list(matrix.shape), 'preprocessing_seconds': preprocessing,
                                     'distance_source': 'existing_cache', 'cache_sources': cache_sources,
                                     'distance_recomputed': False, 'cache_correctness_verified': False,
                                     'timing_scope': 'cache loading, hashing and feature calculation; excludes original distance generation'}
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        simulator = Diffusion(matrix, rivals, cfg['beta'], cfg['horizon'])
        for repeat in cfg.get('repeat_ids', range(cfg['repeats'])):
            # Independent spawned streams; same held-out bank across methods within repeat.
            streams = np.random.SeedSequence([cfg['seed'], repeat]).spawn(3)
            train_seeds = streams[0].generate_state(cfg['train_simulations'], dtype=np.uint64)
            eval_seeds = streams[1].generate_state(cfg['eval_simulations'], dtype=np.uint64)
            learner_seed = int(streams[2].generate_state(1)[0])
            assert not set(train_seeds.tolist()) & set(eval_seeds.tolist())
            np.savez(directory / f'seeds_{repeat}.npz', train=train_seeds, evaluation=eval_seeds)
            for ratio in cfg['ratios']:
                for mode in cfg['modes']:
                    start = time.perf_counter()
                    pool = pool_for(xi, phi, rivals, ratio, cfg['budget'], mode)
                    screening = time.perf_counter() - start
                    for method in cfg['methods']:
                        start = time.perf_counter()
                        training = inference = selection = 0.0
                        checkpoint = None
                        tag = f'{mode}_p{ratio}_{method}_r{repeat}'
                        if method == 'ranking':
                            selected = pool[:cfg['budget']].tolist()
                            selection = time.perf_counter() - start
                        elif method == 'greedy':
                            selected = []
                            for _ in tqdm(range(cfg['budget']), desc='Greedy seeds', unit='seed'):
                                remaining = [int(v) for v in pool if v not in selected]
                                gains = [simulator.samples(selected + [v], train_seeds).mean() for v in tqdm(remaining, desc='Greedy candidates', leave=False)]
                                selected.append(remaining[int(np.argmax(gains))])
                            selection = time.perf_counter() - start
                        elif method in ('dqn', 's2v'):
                            cfg['learner'] = method
                            cfg['checkpoint_path'] = str(directory / (tag + '_training_latest.pt'))
                            cfg['training_log'] = str(directory / (tag + '_training.jsonl'))
                            if cfg.get('resume_directory'):
                                cfg['resume_checkpoint'] = str(Path(cfg['resume_directory']) / name / (tag + '_training_latest.pt'))
                            selected, training, inference, checkpoint = learn(
                                pool, np.column_stack((phi, xi)), cfg['budget'], simulator,
                                train_seeds, cfg, learner_seed)
                        else:
                            raise ValueError(method)
                        assert len(set(selected)) == cfg['budget'] and not set(selected) & set(rivals)
                        start = time.perf_counter()
                        # Persist selected nodes and model BEFORE held-out evaluation.
                        selection_path = directory / (tag + '_selection.json')
                        selection_path.write_text(json.dumps(dict(selected=selected, pool=pool.tolist(),
                            training_seconds=training, inference_seconds=inference, selection_seconds=selection)), encoding='utf-8')
                        if checkpoint is not None:
                            import torch
                            save_checkpoint(directory / (tag + '.pt'), checkpoint)
                        values = []
                        for sample_index, evaluation_seed in enumerate(tqdm(eval_seeds, desc='Final evaluation', unit='simulation')):
                            value = simulator.trial(selected, int(evaluation_seed))
                            append_durable(directory / (tag + '_evaluation.jsonl'),
                                dict(index=sample_index, seed=int(evaluation_seed), influence=value))
                            values.append(value)
                        samples = np.asarray(values)
                        evaluation = time.perf_counter() - start
                        tag = f'{mode}_p{ratio}_{method}_r{repeat}'
                        np.save(directory / (tag + '_samples.npy'), samples)
                        if checkpoint is not None:
                            import torch
                            # Already saved before held-out evaluation.
                            pass
                            manifest['torch'] = torch.__version__
                        record = dict(dataset=name, mode=mode, ratio=ratio, method=method, repeat=repeat,
                                      selected=selected, pool=pool.tolist(), mean=float(samples.mean()),
                                      diffusion_sd=float(samples.std(ddof=1)) if len(samples)>1 else None,
                                      fast_backend=('mixed-reference-fast' if cfg.get('resume_source_backend') == 'reference' else 'choice-bank+reward-memo+sparse-s2v'),
                                      reward_cache_hits=getattr(simulator,'cache_hits',0),
                                      reward_cache_misses=getattr(simulator,'cache_misses',0),
                                      distance_source='existing_cache', distance_generation_seconds=None,
                                      preprocessing_seconds=preprocessing, screening_seconds=screening,
                                      training_seconds=training, inference_seconds=inference,
                                      selection_seconds=selection, evaluation_seconds=evaluation)
                        append_durable(args.output / 'results.jsonl', record)
                        result_path = directory / (tag + '_result.json')
                        temporary = result_path.with_suffix('.tmp')
                        temporary.write_text(json.dumps(record, indent=2), encoding='utf-8')
                        os.replace(temporary, result_path)
                        print(tag, record['mean'], flush=True)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
