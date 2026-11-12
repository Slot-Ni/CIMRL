# HCDP-DQN: Competitive Influence Maximization in Hypergraphs

Code associated with **Structural Prior-Guided Deep Reinforcement Learning for Competitive Influence Maximization in Hypergraphs** (INS-D-26-11982). Historical filenames retain `CSI` and `CSIQ`.

## Main experiment code

| Files | Purpose |
|---|---|
| `CompeteMain.py`, `Run_CSIQ.py`, `CompeteAlgorithm.py`, `CSIQ_Algorithm.py` | Candidate screening and DQN seed selection |
| `Baseline_Main.py`, `Baseline_Algorithm.py` | Structural and greedy baselines |
| `Baseline_Main_S2V.py`, `Baseline_S2V.py` | S2V training and selection |
| `Diffusion_In_HyperGraph.py`, `Indicator.py`, `SeedNodesGenerator.py` | Diffusion, structural scores and seed utilities |
| `Data.py`, `HyperCL.py`, `HighOrderDistance.py` | Input processing, synthetic generation and distances |
| `HCL_evaluate.py`, `HCL1000_Methods_Beta_Comparison.py`, `HCL1000_S2V_Optimized_Comparison.py` | HCL evaluation entry points |
| `plot_all.py`, `Replot_From_JSON.py`, `Result_Image.py`, `Result_Ablation_Image.py`, `Plot_Beta_Ablation_With_S2VOpt.py` | Plotting and ablation figures |

## Installation

```bash
git clone https://github.com/Slot-Ni/CIMRL.git
cd CIMRL
python -m venv .venv
```

Activate `.venv\Scripts\Activate.ps1` on Windows or `source .venv/bin/activate` on Linux/macOS, then:

```bash
python -m pip install -r requirements.txt
```

Dependencies are unpinned. Record the actual Python/package versions used for a run; compatibility with every version is not guaranteed.

## Input layout

```text
workspace/
  CIMRL/
  HyperGraphMatrix/
    Algebra.npy
    Algebra_nodelist.npy
  Hyper_Distance/
    Algebra_distance_1.npy
    ...
    Algebra_distance_10.npy
```

Matrices use nodes as rows and hyperedges as columns. Preserve the node ordering in matrices, mappings and distance arrays. Some scripts use relative paths and others contain original absolute Windows paths, such as `BASE_PATH` in `Run_CSIQ.py`. Inspect and configure the selected entry point's paths before execution. Input archives are available in the repository's data release; they are stored outside Git because of size. Model weights are not included.

## Running main experiments

The entry points are configured in source rather than through a unified CLI. Before running, verify dataset names, seed budgets, candidate ratio, structural order, propagation probabilities, horizon, conflict-resolution rule, simulation counts and output paths in the chosen file.

```bash
python CompeteMain.py
python Baseline_Main.py
python Baseline_Main_S2V.py
python HCL_evaluate.py
```

These are separate jobs, not a mandatory sequential pipeline. Choose the entry point corresponding to the intended experiment. Training can be expensive. Plotting requires the result files expected by the plotting script:

```bash
python plot_all.py
python Plot_Beta_Ablation_With_S2VOpt.py
```

These commands identify entry points; they do not certify that the current defaults reproduce every manuscript table. Check the implementation and retain original logs when reporting model settings or runtime stages. Existing historical caches should not be silently regenerated and treated as unchanged inputs.

## Code version and availability

Record `git rev-parse HEAD` for an exact source version. The initial main-code snapshot is `d4632baf44c5679a68691b2b1e0152a49f99f0b9`. The current branch contains main-experiment code and documentation. Previous Git revisions remain accessible as historical records. No Zenodo DOI is claimed. See CODE_AVAILABILITY.md for release scope.
