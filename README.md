# HCDP-DQN: Competitive Influence Maximization in Hypergraphs

Research code associated with **Structural Prior-Guided Deep Reinforcement Learning for Competitive Influence Maximization in Hypergraphs** (Information Sciences submission INS-D-26-11982).

HCDP-DQN combines competition-aware structural candidate screening with DQN-based sequential seed selection. Legacy filenames retain the names `CSI` and `CSIQ`; these names are preserved for compatibility. Script variants should be matched to the experimental configuration being reproduced.

## What is included

- Baseline methods for node selection
- CSI / CSIQ-style comparison scripts
- S2V-DQN and optimized S2V variants
- Plotting scripts for the main figures and beta ablation results
- Utility code for diffusion, indicators, seed selection, and data loading

## Repository layout

- `Baseline_Algorithm.py`, `Baseline_Main.py`, `Baseline_Main_S2V.py`
  - Baseline selection methods and training entry points
- `CSIQ_Algorithm.py`, `Run_CSIQ.py`
  - CSI candidate selection plus DQN policy
- `HCL_evaluate.py`, `HCL1000_Methods_Beta_Comparison.py`, `HCL1000_S2V_Optimized_Comparison.py`
  - HCL comparison and evaluation scripts
- `plot_all.py`, `Replot_From_JSON.py`, `Result_Image.py`, `New_ablation_ex.py`, `Plot_Beta_Ablation_With_S2VOpt.py`
  - Figure generation scripts
- `Diffusion_In_HyperGraph.py`, `Indicator.py`, `SeedNodesGenerator.py`, `Baseline_S2V.py`
  - Core model and helper modules

## Installation

```bash
git clone https://github.com/Slot-Ni/CIMRL.git
cd CIMRL
python -m venv .venv
```

Activate the environment using `.venv\Scripts\Activate.ps1` on Windows PowerShell or `source .venv/bin/activate` on Linux/macOS.

## Requirements

- Python 3.10 or newer
- `numpy`
- `matplotlib`
- `torch`
- `tqdm`
- `joblib`
- `scikit-learn`
- `pandas`
- `powerlaw`
- `dhg`

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Data

The scripts expect hypergraph matrices and node lists in `HyperGraphMatrix/`, kept beside this `CIM/` folder.
Each dataset should provide:

- `<name>.npy`
- `<name>_nodelist.npy`

Example input layout:

```text
workspace/
├── CIMRL/                         # this repository
├── HyperGraphMatrix/
│   ├── Algebra.npy               # node-by-hyperedge incidence matrix
│   └── Algebra_nodelist.npy       # node IDs in matrix row order
└── Hyper_Distance/               # where required by the chosen script
    └── <name>_distance_<order>.npy
```

Keep node ordering consistent between matrices, node lists, distance arrays, and saved seeds. Some legacy scripts use relative paths and others contain absolute local paths (for example, `BASE_PATH` in `Run_CSIQ.py`). Inspect and update the chosen script's input, cache, and output paths before running. Check the expected cache location for that script; the layout above does not automatically configure every entry point.

Input datasets, distance arrays, trained models, and large generated results are not included in this repository. Source-data references are provided in the manuscript; this checkout alone is not a complete data bundle. Reuse validated existing distance caches when reproducing an existing run; do not silently regenerate or replace them.

## Common commands

```bash
python Main.py
python HCL_evaluate.py
python plot_all.py
python Plot_Beta_Ablation_With_S2VOpt.py
```

Some scripts are expensive to run and reuse cached results when available.

## Code availability

The source code for this project is available in this repository. Large input data and generated result files are excluded from version control because of size, but the corresponding figures and tables can be reproduced from the included scripts when the required datasets are available.

## Experimental configurations and reproducibility

Entry points are research scripts with configuration values in their source, not a single unified command-line application. Inspect the dataset list, propagation rules, seed budget, candidate ratio, random seeds, and simulation counts before execution. The example commands above identify entry points; they do not automatically reproduce every manuscript table.

The main manuscript experiments report 50 Monte Carlo samples for training reward estimation and 10,000 for final influence evaluation. The local revision experiments used a separate configuration: equal transmission probabilities of 0.03, horizon 30, target-priority simultaneous activation, 600 training episodes, 50 reward samples, 1,000 final-evaluation samples, and five independent repetitions. The main manuscript describes competitor-priority activation. These mechanisms and sample budgets must not be conflated.

The revision experiments' portable runners, checkpoint/resume utilities, statistical reports, and additional data are **not currently included in this checkout**. Do not assume that the legacy scripts implement those runners or their exact configuration. In particular, changing a description in the manuscript does not change the behavior of a script.

For reproducible results, record the exact commit, package versions, input files, node ordering, random seeds, diffusion/tie-breaking rules, training budget, and independent repetition count. Report across-repetition standard deviations separately from Monte Carlo variability within a run. Runtime comparisons should state whether distance preprocessing, training, inference, and final evaluation are included, and should identify the hardware used.

## Implementation environment

A recent local revision environment used Windows 11, Intel Core i5-13500, approximately 16 GB RAM, Python 3.13.5, and PyTorch 2.9.0+cpu. This describes the local revision environment, not a verified environment for every legacy script or every historical runtime table. The dependency list is unpinned; record a compatible environment for the entry point being used.

## Citation

If using this code, cite the associated manuscript by its title above. Publication details and a DOI should be added when available.

## Fixed reproducibility release

Revision runners and frozen plans are now included in `revision_experiments/`. Download processed matrices, node mappings, ten-order caches and curated evidence from https://github.com/Slot-Ni/CIMRL/releases/tag/reproducibility-20261111 . See [reproducibility/README.md](reproducibility/README.md) for SHA-256 verification, portable commands, scope and known discrepancies. This supersedes the earlier statement that revision runners are not included. Weights and all historical outputs are not supplied. Cite the full commit linked by the release tag; d4632baf44c5679a68691b2b1e0152a49f99f0b9 identifies the earlier legacy code only.
