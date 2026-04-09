# BasePlaneFittingTerrain

[![CI](https://github.com/chenganggang2020/BasePlaneFittingTerrain/actions/workflows/ci.yml/badge.svg)](https://github.com/chenganggang2020/BasePlaneFittingTerrain/actions/workflows/ci.yml)

Research codebase for robust base-plane fitting, terrain hazard assessment, and safe landing zone evaluation from simulated LiDAR point clouds.

This repository has been reorganized to support:

- structured terrain and outlier simulation
- multiple plane-fitting baselines and robust variants
- risk-map, safety-map, and joint decision analysis
- ablation studies
- paper-style figure reproduction
- one-command end-to-end execution on Windows and Linux/Kylin

## Highlights

- Cross-platform pipeline entrypoint: `run_complete_pipeline.py`
- Parallel data generation and experiment execution
- Structured contamination scenarios for better evaluation of `RSTLS`
- Standard-library-first result aggregation path for easier deployment
- GitHub-ready repository layout with CI and ignore rules

## Repository Layout

- `run_complete_pipeline.py`
  One-command pipeline runner for simulation, main experiments, ablation, and paper-style reporting.
- `run_paper_experiments.py`
  Runs the main experiment suite for a selected preset.
- `run_ablation_study.py`
  Runs the ablation study suite.
- `reproduce_paper_figures.py`
  Regenerates paper-style figures and summary tables from existing results.
- `OutlierTerrainSimulator.py`
  Structured simulator with stripe noise, bad scan batches, and distance-dependent heteroscedastic noise.
- `monte_carlo.py`
  Main batch analyzer and result aggregation pipeline.
- `algorithm_runner.py`
  Algorithm registry for baseline and robust fitting methods.

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
python run_complete_pipeline.py --preset structured35 --workers auto --generation-workers auto --force-reprocess --no-save-vector --dpi 300
```

Linux / Kylin:

```bash
python3 run_complete_pipeline.py --preset structured35 --workers auto --generation-workers auto --force-reprocess --no-save-vector --dpi 300
```

### Run a fast smoke-test version

```powershell
python run_complete_pipeline.py --preset structured35 --workers auto --generation-workers auto --force-reprocess --no-save-vector --dpi 150 --skip-ablation --skip-paper-report --include-methods RANSAC,RobustLS-TLS,RSTLS --disable-risk --disable-safety --disable-joint
```

## Common Commands

Generate structured simulation data only:

```powershell
python OutlierTerrainSimulator.py --workers auto
```

Run the main experiment only:

```powershell
python run_paper_experiments.py --preset structured35 --workers auto --force-reprocess --no-save-vector --dpi 300
```

Run ablation only:

```powershell
python run_ablation_study.py --preset structured35 --ablation all --workers auto
```

Reproduce paper-style figures from existing outputs:

```powershell
python reproduce_paper_figures.py --preset structured35
```

## Output Layout

Typical generated content includes:

- simulated DEM and point-cloud folders
- experiment result directories under `results/`
- global comparison plots
- paper-style reproduced figures and summary markdown reports

Generated data and result folders are intentionally ignored by Git so the repository stays lightweight.

## Recommended Workflow

1. Run a small smoke test first.
2. Run the full preset once the pipeline is stable.
3. Reproduce the paper-style figures from the generated result directory.
4. Run ablations only after the main experiment output looks correct.

See [CONTRIBUTING.md](./CONTRIBUTING.md) for a simple branch and commit workflow.

## GitHub Sync Notes

The repository ignores the following by default:

- generated point clouds
- generated DEMs
- `results/`
- `_tmp*` debugging folders

For large result sharing, prefer:

- GitHub Releases
- Git LFS
- external object storage or cloud drive links

## Current Status

- CI checks basic syntax and entrypoint health
- the main pipeline supports Windows and Linux/Kylin command-line usage
- parallel execution is available for both data generation and experiment runs

## Citation / Usage

If you use this repository in a paper or report, please cite the corresponding method description and clearly state the simulation preset and algorithm subset used in your experiments.
