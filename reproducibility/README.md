# Reproducibility release

Fixed tag: `reproducibility-20261111`. Original legacy commit: `d4632baf44c5679a68691b2b1e0152a49f99f0b9`. Cite the full commit linked by the new release for this expanded snapshot.

Download assets: https://github.com/Slot-Ni/CIMRL/releases/tag/reproducibility-20261111

- processed_hypergraphs.zip: nine main real networks, nine main HCL networks, senate-committees and high-school_2012; original matrices and node lists.
- existing_distance_caches.zip: the existing ten-order distance arrays for those networks.
- revision_result_evidence.zip: curated result records, held-out evaluation samples, simulation seed banks, manifests and receipts from local revision runs and the Bars-Rev 34-task snapshot.

Extract data/cache archives beside the repository into `../HyperGraphMatrix/` and `../Hyper_Distance/`. Preserve node ordering. Verify archive SHA-256 values in release_assets.json and per-input values in input_manifest.json.

## Portable commands

Install numpy, scipy, torch and tqdm in a compatible environment; environment-local.txt captures the recent local package versions, not a universal dependency compatibility guarantee. From repository root:

```bash
python revision_experiments/revision_C04_real_extension_20261109/run_suite.py check --plan revision_experiments/revision_C04_real_extension_20261109/plan.json --data-root ../HyperGraphMatrix --cache-root ../Hyper_Distance
python revision_experiments/revision_C04_real_extension_20261109/run_suite.py run --plan revision_experiments/revision_C04_real_extension_20261109/plan.json --data-root ../HyperGraphMatrix --cache-root ../Hyper_Distance --output reproduced_results/c04_real --workers 3 --s2v-workers 1
```

Source hashes are frozen in plans; do not edit Python files for an existing plan. Old PowerShell/Notebook convenience launchers retain original machine-specific paths: use the explicit paths above when migrating. On Windows the runner's .ops directory can be added to PYTHONPATH for checkpoint replacement retries. Same-plan reruns skip verified completed tasks and resume per-episode DQN/S2V checkpoints. Never launch two schedulers into one output directory.

## Evidence scope and limitations

Model weights and bulky intermediate features are omitted from the evidence archive. Receipts retain original hashes, including omitted artifacts; this curated evidence is not a resumable output tree and full receipt verification requires the omitted files. Use fresh output directories for reruns. Some tasks appear in multiple folders through adoption; do not count duplicates as independent repetitions.

Algebra shared-pool results have five repetitions per method. Bars-Rev has five DQN, five Greedy and four S2V repetitions. Revision final evaluation uses 1000 simulations, not 10000, with 600 training episodes and 50 reward samples. Revision propagation uses beta=.03 for both entities, T=30 and target-priority ties. Main manuscript experiments describe competitor-priority ties. Publishing these files does not resolve this mechanism discrepancy.

Caches are preserved historical inputs, not certified shortest-path truth. Legacy HCL cache generation has a known source-row omission issue; do not silently repair and claim identical reproduction. Main historical model checkpoints and every original figure's raw outputs are not supplied. Cross-machine training times are descriptive and not controlled hardware speed benchmarks. Retain null and negative results alongside favorable ones.
