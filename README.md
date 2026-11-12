# HCDP-DQN: Competitive Influence Maximization in Hypergraphs

Code for **Structural Prior-Guided Deep Reinforcement Learning for Competitive Influence Maximization in Hypergraphs** (INS-D-26-11982).

## Main experiment entry point

`Run_CSIQ.py` is the single entry point for the HCDP-DQN main experiments. `CompeteMain.py` has been removed. Use `--budget N` to explicitly select the target budget for a figure; otherwise the historical 1% convention applies.

```bash
# All 9 real-world and 9 synthetic datasets
python Run_CSIQ.py --group all

# Real-world datasets only (beta = 0.025 and 0.05)
python Run_CSIQ.py --group real

# Synthetic datasets only (beta = 0.025)
python Run_CSIQ.py --group synthetic

# One dataset
python Run_CSIQ.py --dataset Algebra
```

The main-run defaults are centralized in `PAPER_CONFIG` in `Run_CSIQ.py`:

| Setting | Main-paper value |
|---|---:|
| Propagation horizon | 30 |
| Real-world common transmission probabilities | 0.025, 0.05 |
| Synthetic common transmission probability | 0.025 |
| Reward-estimation simulations | 50 |
| Final evaluation simulations | 10,000 |
| Candidate ratio | 0.20 |
| Maximum structural order | 10 |
| Order-weight parameter | 0.5 |
| Learning rate | 0.03 |
| Discount factor | 0.9 |
| Training episodes | 600 |
| Replay-buffer capacity | 30,000 |
| Mini-batch size | 64 |
| Target-network update interval | 100 decision steps |
| Exploration schedule | 0.5 to 0.01, linear |
| Q-network hidden widths | 256, 128 |

The runner follows the repository's existing main-run seed-budget convention of selecting 1% of the nodes as competing seeds and target budget. Check the manuscript's precise budget definition before changing this convention.

## Other experiment scripts

The following are separate supplementary or baseline workflows, not replacements for the main entry point:

- `Baseline_Main.py`: structural and greedy baselines.
- `Baseline_Main_S2V.py`: full-feasible-set S2V-DQN baseline.
- `HCL_evaluate.py` and `HCL1000_*.py`: supplementary HCL comparisons and evaluations.
- `Beta_Ablation_CSIQ.py`: propagation-probability sensitivity.
- Plotting scripts: render figures from previously generated result files.

Supplementary experiments have their own settings in the manuscript. Do not assume that their training or evaluation counts are identical to the main experiments.

## Input data layout

The runner expects the following sibling-folder layout:

```text
workspace/
  CIMRL/
    Run_CSIQ.py
    ...
  HyperGraphMatrix/
    Algebra.npy
    Algebra_nodelist.npy
    ...
  Hyper_Distance/
    Algebra_distance_1.npy
    ...
    Algebra_distance_10.npy
```

Matrices store nodes in rows and hyperedges in columns. The node ordering must be identical across the incidence matrix, node list, and higher-order distance arrays. The distance files for all ten orders must be present.

## Installation

```bash
git clone https://github.com/Slot-Ni/CIMRL.git
cd CIMRL
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate

python -m pip install -r requirements.txt
```

Dependencies are currently not pinned. For reproducibility, record the Python, NumPy, PyTorch, and other package versions used.

## Outputs

Main-run JSON files are written to `results/main_hcdp_dqn/`. Each file records the dataset, beta, selected nodes, influence values for each selected-set prefix, runtime, and configuration.

## Reproducibility note

Parameter alignment does not by itself establish full algorithmic equivalence. In particular, verify the simultaneous conflict-resolution implementation in `Diffusion_In_HyperGraph.py` and the next-action masking in `CompeteAlgorithm.py` against the paper's ACHPD definition and Algorithm 3 before treating this repository revision as a complete reproduction. Preserve the exact Git commit and raw experiment logs used for every reported table and figure.

## Verification and provenance

This update changes executable behavior, including simultaneous conflict resolution and masked next-action values. It does not regenerate or validate historical manuscript numbers. Run `python -m unittest test_paper_alignment -v` for small deterministic checks. No full-network training was rerun during this update.

The main runner uses disjoint fixed seed ranges for training rewards (starting42) and final evaluation (starting1000000). Network inputs now follow the paper order [xi, phi]. Old checkpoints are not compatible evidence for this changed training implementation.

The other historical experiment/plot scripts may retain their original workflow-specific parameters. They are not certified globally equivalent to the paper merely because main-run defaults were centralized. The supplied baseline S2V learner requires separate architectural/masking review; main runner alignment must not be generalized to every legacy entry point. Main hyperparameter transcription does not establish the provenance of old runtime tables.
