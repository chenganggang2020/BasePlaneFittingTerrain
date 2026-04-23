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
python code/run_complete_pipeline.py --preset structured35 --pipeline-mode streaming --workers auto --simulator-profile landing --base-seed 42 --force-reprocess --no-save-vector --dpi 300
```

Linux / Kylin:

```bash
python3 code/run_complete_pipeline.py --preset structured35 --pipeline-mode streaming --workers auto --simulator-profile landing --base-seed 42 --force-reprocess --no-save-vector --dpi 300
```

### Run a fast smoke-test version

```powershell
python code/run_complete_pipeline.py --preset structured35 --pipeline-mode streaming --workers auto --simulator-profile landing --base-seed 42 --force-reprocess --no-save-vector --dpi 150 --skip-ablation --skip-paper-report --include-methods RANSAC,RobustLS-TLS,RSTLS --disable-risk --disable-safety --disable-joint
```

### Run from a reusable config file

```powershell
python code/run_complete_pipeline.py --config configs/pipeline_structured35.json
```

```powershell
python code/run_complete_pipeline.py --config configs/pipeline_smoke.json
```

## Common Commands

Generate structured simulation data only:

```powershell
python code/OutlierTerrainSimulator.py --profile landing --base-seed 42 --workers auto
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

Config templates:

- `configs/pipeline_structured35.json`
- `configs/pipeline_smoke.json`
- `configs/README.md`

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
- parallel execution is available for data generation, point-cloud solving, and streaming generate-then-solve workflows
- code, data, results, paper, and legacy scripts are now separated at the folder level

## Parallel and Reproducibility

- `--workers auto`
  Uses CPU-count-based parallelism for point-cloud solving or streaming generate+solve mode.
- `--generation-workers auto`
  Uses separate worker control for staged generation mode.
- `--pipeline-mode streaming`
  Recommended. Each worker generates one outlier-ratio dataset and immediately runs the scan/solution pipeline for it.
- `--simulator-profile landing`
  Recommended paper-oriented simulator profile with a smoother candidate landing zone, clustered hazards, and structured artefacts.
- `--base-seed 42`
  Keeps the synthetic dataset reproducible across Windows and Linux/Kylin.
- `--force-regenerate`
  Rebuilds synthetic DEM and point-cloud files even if matching outputs already exist.

## Simulation Realism

- The structured simulator now combines macro relief, micro relief, clustered hazards, stripe corruption, and bad scan batches.
- The exported point cloud is no longer a full regular grid by default; it is thinned by a LiDAR-like observation model that depends on sensor position, incidence angle, range attenuation, and edge/shadow dropouts.
- Each generated point cloud writes a sidecar metadata JSON such as `point_cloud_10pct_metadata.json` with the nominal ratio, observed ratio, kept-point count, and sensor model parameters.
- Recommended profile:
  `landing`
  This profile keeps a smoother candidate landing zone near the center while biasing hazards toward off-center clustered regions.

## Citation / Usage

If you use this repository in a paper or report, please cite the corresponding method description and clearly state the simulation preset and algorithm subset used in your experiments.
