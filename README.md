# BasePlaneFittingTerrain

[![CI](https://github.com/chenganggang2020/BasePlaneFittingTerrain/actions/workflows/ci.yml/badge.svg)](https://github.com/chenganggang2020/BasePlaneFittingTerrain/actions/workflows/ci.yml)

Research codebase for robust base-plane fitting, terrain hazard assessment, and safe landing zone evaluation from simulated LiDAR point clouds.

## Repository Layout

The project is now organized by purpose rather than by script age:

- `code/`
  Main Python source code and runnable entry scripts.
- `data/`
  Generated DEM and point-cloud datasets.
- `results/`
  Experiment outputs, split into `paper/`, `debug/`, and `archive/`.
- `paper/`
  Paper materials, manuscript drafts, extracted text, and method notes.
- `docs/`
  Project-facing technical notes and repository status documents.
- `legacy/`
  Older experimental entry scripts kept only for reference/compatibility.

## Project Status Docs

- Chinese project status and paper-comparison note:
  `docs/Current_Code_Results_and_Paper_Comparison_CN.md`

## Main Entry Points

Run these from the repository root:

- `code/run_complete_pipeline.py`
  One-command pipeline runner for simulation, main experiments, ablation, and paper-style reporting.
- `code/run_paper_experiments.py`
  Runs the main experiment suite for a selected preset.
- `code/run_ablation_study.py`
  Runs the ablation study suite.
- `code/reproduce_paper_figures.py`
  Regenerates paper-style figures and summary tables from existing results.
- `code/OutlierTerrainSimulator.py`
  Structured simulator with stripe noise, bad scan batches, and distance-dependent heteroscedastic noise.

## Installation

Core dependencies:

```bash
pip install -r requirements.txt
```

Optional dependencies:

- `GDAL / osgeo`
  Enables GeoTIFF export. If unavailable, DEM export falls back to `.npy`.
- `pandas`
  No longer required for the main pipeline, but still supported by a few legacy visualization paths.
- `seaborn`
  Used only by selected plotting utilities when available.

## Quick Start

### Run the complete pipeline

Windows:

```powershell
python code/run_complete_pipeline.py --preset structured35 --workers auto --generation-workers auto --force-reprocess --no-save-vector --dpi 300
```

Linux / Kylin:

```bash
python3 code/run_complete_pipeline.py --preset structured35 --workers auto --generation-workers auto --force-reprocess --no-save-vector --dpi 300
```

### Run a fast smoke-test version

```powershell
python code/run_complete_pipeline.py --preset structured35 --workers auto --generation-workers auto --force-reprocess --no-save-vector --dpi 150 --skip-ablation --skip-paper-report --include-methods RANSAC,RobustLS-TLS,RSTLS --disable-risk --disable-safety --disable-joint
```

## Common Commands

Generate structured simulation data only:

```powershell
python code/OutlierTerrainSimulator.py --workers auto
```

Run the main experiment only:

```powershell
python code/run_paper_experiments.py --preset structured35 --workers auto --force-reprocess --no-save-vector --dpi 300
```

Run ablation only:

```powershell
python code/run_ablation_study.py --preset structured35 --ablation all --workers auto
```

Reproduce paper-style figures from existing outputs:

```powershell
python code/reproduce_paper_figures.py --preset structured35
```

## Data and Paper Organization

- Put generated or copied point-cloud/DEM folders under `data/`.
- Put manuscript drafts, PDF, DOCX, and external paper files under `paper/original/`.
- Put extracted text, reading notes, and method-gap notes under `paper/notes/`.
- Put formal experiment outputs under `results/paper/`.
- Put temporary or ad-hoc outputs under `results/debug/`.
- Move old result trees that should be kept but not reused into `results/archive/`.

See:

- `data/README.md`
- `paper/README.md`

## GitHub Sync Notes

The repository ignores generated datasets and experiment outputs by default:

- generated DEM folders
- generated point-cloud folders
- `results/`
- `_tmp*` debugging files

For large result sharing, prefer:

- GitHub Releases
- Git LFS
- external object storage or cloud drive links

## Current Status

- CI checks syntax and entrypoint health
- the main pipeline supports Windows and Linux/Kylin command-line usage
- parallel execution is available for both data generation and experiment runs
- code, data, results, paper, and legacy scripts are now separated at the folder level

## Citation / Usage

If you use this repository in a paper or report, please cite the corresponding method description and clearly state the simulation preset and algorithm subset used in your experiments.
