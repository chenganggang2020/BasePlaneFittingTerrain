# BasePlaneFittingTerrain

基于激光雷达点云的基准面拟合、障碍检测与安全着陆评估实验项目。

这个仓库现在已经整理成适合直接同步到 GitHub 的研究型项目结构，包含：
- 结构化模拟数据生成
- 多种平面拟合算法对比
- 风险图 / 安全图 / 联合决策评估
- 消融实验
- 论文风格图表复现
- 一键式完整流程入口

## 目录说明

- `run_complete_pipeline.py`
  一键总控脚本，按顺序完成数据生成、主实验、消融实验和论文图表复现。
- `run_paper_experiments.py`
  跑主实验与全局对比图。
- `run_ablation_study.py`
  跑消融实验。
- `reproduce_paper_figures.py`
  基于已有结果目录重绘论文风格图表和表格。
- `OutlierTerrainSimulator.py`
  结构化模拟器，支持条带噪声、连续坏批次和距离相关异方差噪声。
- `monte_carlo.py`
  主分析器与批处理流程，已支持点云级并行。

## 环境要求

核心依赖：

```bash
pip install -r requirements.txt
```

可选依赖：
- `GDAL / osgeo`
  用于导出 GeoTIFF；如果没有安装，模拟器会自动回退为 `.npy` DEM 输出。
- `pandas`
  当前主流程不依赖 `pandas`，但个别历史可视化分支仍保留可选兼容。
- `seaborn`
  仅部分可视化函数会尝试按需导入。

## 快速开始

### 1. 一键跑完整流程

Windows:

```powershell
python run_complete_pipeline.py --preset structured35 --workers auto --generation-workers auto --force-reprocess --no-save-vector --dpi 300
```

Linux / 麒麟:

```bash
python3 run_complete_pipeline.py --preset structured35 --workers auto --generation-workers auto --force-reprocess --no-save-vector --dpi 300
```

### 2. 先跑一个快速验证版

```powershell
python run_complete_pipeline.py --preset structured35 --workers auto --generation-workers auto --force-reprocess --no-save-vector --dpi 150 --skip-ablation --skip-paper-report --include-methods RANSAC,RobustLS-TLS,RSTLS --disable-risk --disable-safety --disable-joint
```

## 常用脚本

只生成结构化模拟数据：

```powershell
python OutlierTerrainSimulator.py --workers auto
```

只跑主实验：

```powershell
python run_paper_experiments.py --preset structured35 --workers auto --force-reprocess --no-save-vector --dpi 300
```

只跑消融：

```powershell
python run_ablation_study.py --preset structured35 --ablation all --workers auto
```

只重绘论文图表：

```powershell
python reproduce_paper_figures.py --preset structured35
```

## GitHub 同步建议

本仓库默认不会把以下内容提交到 GitHub：
- 生成点云
- 生成 DEM
- `results/` 实验结果
- `_tmp*` 临时调试目录

这样仓库会保持轻量，便于同步和协作。如果你确实需要分享大结果文件，建议使用：
- GitHub Releases
- Git LFS
- 网盘 / 对象存储

## 初始化并推送到 GitHub

如果本地还没有远端仓库，可以按下面流程：

```bash
git init -b main
git add .
git commit -m "Initial research pipeline setup"
git remote add origin <your-github-repo-url>
git push -u origin main
```

## 说明

- 已为跨平台运行整理了无界面 `Agg` 绘图后端。
- 受限环境下如果进程池不可用，会自动回退到线程池。
- GitHub Actions 已提供最小 CI，用于做语法和入口健康检查。
