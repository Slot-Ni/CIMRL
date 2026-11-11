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

The revision experiments' portable runners and frozen plans are now included under `revision_experiments/`; data and curated evidence are published as release assets. Do not assume that the legacy scripts implement those runners or their exact configuration. In particular, changing a description in the manuscript does not change the behavior of a script.

For reproducible results, record the exact commit, package versions, input files, node ordering, random seeds, diffusion/tie-breaking rules, training budget, and independent repetition count. Report across-repetition standard deviations separately from Monte Carlo variability within a run. Runtime comparisons should state whether distance preprocessing, training, inference, and final evaluation are included, and should identify the hardware used.

## Implementation environment

A recent local revision environment used Windows 11, Intel Core i5-13500, approximately 16 GB RAM, Python 3.13.5, and PyTorch 2.9.0+cpu. This describes the local revision environment, not a verified environment for every legacy script or every historical runtime table. The dependency list is unpinned; record a compatible environment for the entry point being used.

## Citation

If using this code, cite the associated manuscript by its title above. Publication details and a DOI should be added when available.

## Fixed reproducibility release

Revision runners and frozen plans are now included in `revision_experiments/`. Download processed matrices, node mappings, ten-order caches and curated evidence from https://github.com/Slot-Ni/CIMRL/releases/tag/reproducibility-20261111 . See [reproducibility/README.md](reproducibility/README.md) for SHA-256 verification, portable commands, scope and known discrepancies. This supersedes the earlier statement that revision runners are not included. Weights and all historical outputs are not supplied. Cite the full commit linked by the release tag; d4632baf44c5679a68691b2b1e0152a49f99f0b9 identifies the earlier legacy code only.

## Reproduce the revision experiments

1. Obtain the fixed source version: `git checkout reproducibility-20261111` (published baseline); the newer `reproducibility-guide-20261111` tag adds the portable dispatcher documented below. Use the latter for these commands and record `git rev-parse HEAD`.
2. Download all three ZIP files from the [data release](https://github.com/Slot-Ni/CIMRL/releases/tag/reproducibility-20261111).
3. Run `python reproducibility/verify_downloads.py /path/to/downloads` before extracting. Extract matrices and caches next to the repository as described in the guide.
4. Install `numpy scipy torch tqdm` for revision runners. Legacy plotting scripts additionally use requirements.txt. See the captured environment and limitations in the guide.
5. Validate the chosen plan, then explicitly request training:

```bash
python reproducibility/run_experiment.py c04_real_extension
python reproducibility/run_experiment.py c04_real_extension --run --workers 3
python reproducibility/run_experiment.py c04_real_extension --merge
```

The first command only checks source/input hashes. `--run` starts training. `--merge` audits all completed task receipts and writes merged tables alongside the output folder; incomplete merges fail instead of silently treating missing tasks as zero. Default output: `reproduced_results/<experiment>/`. Override data/cache paths with `--data-root` and `--cache-root`; override output with `--output`. Logs and checkpoint files are inside task-ID folders.

| Experiment key | Scope | Planned tasks |
|---|---|---:|
| c04_original | Historical combined C04/C05/C07 plan on Algebra/HCL; includes expensive S2V, not only sensitivity | 140 |
| c04_real_extension | Bars-Rev ratios .1/.4 and NDC five ratios, DQN | 35 |
| c05_order_pair | First-order screening on Algebra/HCL; ten-order references are earlier runs | 10 |
| c05_order_extension | Both orders on senate-committees, Restaurants-Rev, Geometry | 30 |
| c05_order_round2 | Both orders on Music-Rev, NDC-classes, high-school_2012 | 30 |
| c07_algebra | Shared-pool DQN/Greedy/S2V, five repetitions each | 15 |

Use the same command/output to resume. Do not edit frozen source files or regenerate historical caches. A new full run can take many hours or days, especially S2V. Existing result evidence can be inspected without rerunning training; it deliberately omits model weights and intermediate features, so it cannot itself serve as a complete resumable output directory. For C05 paired analysis, the first-order tasks and historical ten-order references must both be retained. Bars-Rev's earlier ratios are in the server evidence, not among the 35 new tasks.

### Configuration used by revision runners

Product screening; default ratio .2 (C04 varies .05/.1/.2/.4/1); target budget10; orders10; alpha.5; equal beta.03; T30; target-priority ties; 600 episodes; 50 reward samples; 1000 held-out evaluation samples; five repeats. S2V uses32-dimensional embeddings and32 message-passing rounds. Full-space legacy S2V and same-pool revision S2V are distinct comparisons. Do not describe these samples as ten repetitions or 10000 evaluations.

### Fixed citation

Use a commit permalink, not only the moving main branch. The data-bearing baseline is commit `f91abd577e731faaf853ddbab5ba8a6e30370cad`; the `reproducibility-guide-20261111` release fixes the expanded execution guide and its source commit. No Zenodo DOI has been minted or claimed. See the release's linked commit for the exact newer hash.
