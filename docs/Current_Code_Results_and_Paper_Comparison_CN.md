# 当前代码整理、结果状态与论文对照说明

## 1. 文档目的

这份文档主要回答 3 个问题：

1. 当前仓库里哪些代码是主线，应该从哪里开始运行。
2. 当前仓库里已经生成了哪些结果，哪些结果可以直接引用，哪些还只是历史残留。
3. 当前实现与论文《基于激光雷达的深空软着陆稳健障碍检测方法研究》之间，已经对齐到什么程度，还差什么。

当前对照主要基于：

- [论文抽取文本](D:/BasePlaneFittingTerrain/paper/notes/extracted_text/_tmp_paper.txt)
- [主实验汇总表](D:/BasePlaneFittingTerrain/results/paper/legacy_root_summary/global_comparison_plots/global_aggregated_results.csv)
- [主流程入口](D:/BasePlaneFittingTerrain/code/run_complete_pipeline.py)
- [实验预设](D:/BasePlaneFittingTerrain/code/experiment_presets.py)
- [实验主调度](D:/BasePlaneFittingTerrain/code/monte_carlo.py)

## 2. 当前仓库结构

当前仓库已经按用途拆成 6 层：

- `code/`
  主代码、算法实现、实验入口。
- `data/`
  DEM、点云和模拟数据。
- `results/`
  实验结果、图表和 manifest。
- `paper/`
  论文原稿、抽取文本和方法笔记。
- `docs/`
  面向项目维护的说明文档。
- `legacy/`
  旧入口脚本和兼容脚本。

### 2.1 当前最推荐使用的入口

现在建议只把下面 4 个脚本视为主入口：

- [run_complete_pipeline.py](D:/BasePlaneFittingTerrain/code/run_complete_pipeline.py)
- [run_paper_experiments.py](D:/BasePlaneFittingTerrain/code/run_paper_experiments.py)
- [run_ablation_study.py](D:/BasePlaneFittingTerrain/code/run_ablation_study.py)
- [reproduce_paper_figures.py](D:/BasePlaneFittingTerrain/code/reproduce_paper_figures.py)

### 2.2 当前主线核心模块

| 逻辑分组 | 主要文件 | 当前作用 |
| --- | --- | --- |
| 数据生成 | `code/OutlierTerrainSimulator.py` | 生成 DEM、点云、结构化污染场景 |
| 算法注册 | `code/algorithm_runner.py` | 统一注册 LS / RANSAC / RWTLS / RobustLS-TLS / RSTLS |
| 拟合算法 | `code/ls.py`、`code/ransac.py`、`code/lmedsq.py`、`code/mad.py`、`code/pdimse.py`、`code/ikose.py`、`code/rwtls.py`、`code/robust_ls_tls.py`、`code/sequential_robust_tls.py` | 各类平面拟合或稳健拟合器 |
| 风险/安全评价 | `code/terrain_features.py`、`code/risk_probability_map.py`、`code/safety_evaluation.py`、`code/joint_risk_safety.py` | 风险图、安全系数和联合决策 |
| 实验主流程 | `code/monte_carlo.py`、`code/experiment_presets.py` | 批量实验调度、结果汇总、预设管理 |
| 可视化与表格 | `code/visualization.py`、`code/visualizer.py`、`code/tabular_results.py` | 出图、CSV 汇总、论文图表复现 |

### 2.3 当前属于历史兼容的脚本

以下脚本已经移到 `legacy/`，后续不建议再继续扩展：

- `legacy/MultiOutlierRatio_Experiment.py`
- `legacy/MultiOutlierRatio_Experiment-CGG.py`
- `legacy/MultiOutlierRatio_Experiment-CGG-2.py`
- `legacy/MultiOutlierRatio_Experiment-CGG-3.py`
- `legacy/OutlierTerrainSimulator-CGG.py`
- `legacy/algorithm_runner-CGG.py`
- `legacy/sequential_robust_tls-CGG.py`

## 3. 当前主流程代码结构

### 3.1 关键位置

- [monte_carlo.py](D:/BasePlaneFittingTerrain/code/monte_carlo.py):210
  `PointCloudAnalyzer` 主类。
- [monte_carlo.py](D:/BasePlaneFittingTerrain/code/monte_carlo.py):808
  `analyze_point_cloud()`，负责单个点云分析。
- [monte_carlo.py](D:/BasePlaneFittingTerrain/code/monte_carlo.py):988
  `run_outlier_sweep()`，负责离群点比例扫描。
- [algorithm_runner.py](D:/BasePlaneFittingTerrain/code/algorithm_runner.py):30
  `get_algorithm_categories()`，负责算法注册与筛选。
- [experiment_presets.py](D:/BasePlaneFittingTerrain/code/experiment_presets.py):82
  `PAPER_EXPERIMENT_PRESETS`
- [experiment_presets.py](D:/BasePlaneFittingTerrain/code/experiment_presets.py):115
  `ABLATION_CASES`

### 3.2 已接入的论文相关模块

相较更早的版本，当前仓库已经补上了这些关键模块：

- [robust_ls_tls.py](D:/BasePlaneFittingTerrain/code/robust_ls_tls.py):4
  `RobustLSTLSPlaneFitter`
- [sequential_robust_tls.py](D:/BasePlaneFittingTerrain/code/sequential_robust_tls.py):4
  `SequentialRobustTLSPlaneFitter`
- [terrain_features.py](D:/BasePlaneFittingTerrain/code/terrain_features.py):52
  地形特征提取
- [risk_probability_map.py](D:/BasePlaneFittingTerrain/code/risk_probability_map.py):123
  风险概率图生成
- [safety_evaluation.py](D:/BasePlaneFittingTerrain/code/safety_evaluation.py):186
  安全系数评价
- [joint_risk_safety.py](D:/BasePlaneFittingTerrain/code/joint_risk_safety.py):108
  风险图与安全图联合决策

## 4. 当前结果目录的真实状态

### 4.1 当前最可靠的结果表

目前最稳定、最完整、最适合直接写进说明文档的结果表是：

- [global_aggregated_results.csv](D:/BasePlaneFittingTerrain/results/paper/legacy_root_summary/global_comparison_plots/global_aggregated_results.csv)

这张表的特点是：

- 共 `648` 行
- 对应 `81` 个离群点比例
- 共 `8` 个方法
- 当前稳定对应的是“随机离群点 sweep”这条实验线

当前表中实际出现的方法只有：

- `MAD`
- `LMedsq`
- `RANSAC`
- `IKOSE`
- `PDIMSE`
- `LS`
- `RWTLS`
- `RSTLS`

也就是说：

- 当前代码里虽然已经接入了 `RobustLS-TLS`
- 但当前正式沉淀下来的主结果表，还没有把它完整写进去

### 4.2 当前结果树的问题

仓库里仍然保留着多套历史结果目录，例如：

- `results/paper/legacy_random_sweep/`
- `results/archive/outlier_sweep_results0/`
- `results/archive/outlier_sweep_results1/`
- `results/archive/outlier_sweep_results5/`
- `results/archive/outlier_sweep_results10/`
- `results/archive/outlier_sweep_results_root/`

这说明仓库经历了多轮实验演化，但也意味着：

- 当前结果树仍然偏乱
- “正式可引用结果”和“调试残留结果”还没有完全分开

建议后续统一约定：

- 正式结果：`results/paper_*`
- 调试结果：`results/debug_*`
- 归档结果：`results/archive/*`

### 4.3 当前结构化数据的状态

当前结构化场景数据已经集中到 `data/` 下，例如：

- `data/dem_files_output_structured35/`
- `data/point_clouds_output_structured35/`

但当前正式结果目录里还没有稳定沉淀出：

- `results/paper/structured35/`

这意味着：

- 结构化污染数据已经具备
- 但还没有把最能体现 `RSTLS` 优势的正式实验结果跑完整并整理成标准产出

### 4.4 当前 manifest 的状态

[paper_experiment_manifest.json](D:/BasePlaneFittingTerrain/results/paper/manifests/paper_experiment_manifest.json) 里如果继续使用旧结果，仍可能保留旧的 `D:\OneDrive\python\BasePlaneFittingTerrain` 路径。

这不是代码逻辑错误，而是说明：

- 仓库迁移到 `D:\BasePlaneFittingTerrain` 后
- 这份 manifest 还没有通过新路径重跑一次来刷新

## 5. 当前已生成结果的摘要分析

### 5.1 平均性能概览

以下统计基于 [global_aggregated_results.csv](D:/BasePlaneFittingTerrain/results/paper/legacy_root_summary/global_comparison_plots/global_aggregated_results.csv)：

| Method | Mean Accuracy | Accuracy 10%-40% | Accuracy 40%-90% | Mean FPR | Mean FNR | Mean CPU (s) | Mean Normal Deviation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RANSAC | 0.9849 | 0.9855 | 0.9849 | 0.0054 | 0.0445 | 0.0055 | 0.0956 |
| RSTLS | 0.9437 | 0.9998 | 0.9107 | 0.0163 | 0.1948 | 0.0137 | 1.7219 |
| LMedsq | 0.7992 | 0.8893 | 0.7476 | 0.2095 | 0.0637 | 0.0872 | 0.8888 |
| RWTLS | 0.7621 | 0.8918 | 0.6852 | 0.3794 | 0.2159 | 22.8047 | 3.6220 |
| IKOSE | 0.7422 | 0.9768 | 0.6040 | 0.3251 | 0.0230 | 0.0109 | 1.5715 |
| PDIMSE | 0.6794 | 0.9680 | 0.5098 | 0.4807 | 0.0283 | 0.0211 | 2.6264 |
| MAD | 0.6721 | 0.9851 | 0.4878 | 0.4622 | 0.0090 | 0.0011 | 2.2050 |
| LS | 0.5005 | 0.7505 | 0.3505 | 1.0000 | 0.0000 | 0.0010 | 3.9155 |

### 5.2 当前结果最直接的结论

在当前“随机离群点 sweep”结果里，可以直接得出：

1. `RANSAC` 仍然是当前已沉淀结果里最强的总体基线。
2. `RSTLS` 是当前结果表里的第二梯队最强方法，尤其在 `10%-40%` 区间几乎达到满分。
3. `RSTLS` 在高离群率区间仍然较稳，但总体均值还没有超过 `RANSAC`。
4. `RWTLS` 运行时间过大，工程性最弱。
5. `LS` 只能作为最基础参照，不适合作为最终方案。

### 5.3 为什么当前结果还不能充分体现 RSTLS 的优势

当前结果还不能充分体现 `RSTLS` 的优势，原因不是“代码完全无效”，而是“正式实验场景和目标不够匹配”。

更具体地说：

- 当前正式结果主要还是随机离群点扫描
- 这类场景天然更有利于 `RANSAC` 这类硬剔除方法
- 而 `RSTLS` 更适合展示在：
  - 结构化污染
  - 连续坏批次
  - 距离相关异方差
  - 序列更新
  - 安全区搜索联动

换句话说：

> 当前代码方向已经比老版本更适合 `RSTLS`，但正式沉淀下来的结果还没有完全跟上。

## 6. 与论文的逐项对照

### 6.1 论文方法主线

根据 [论文抽取文本](D:/BasePlaneFittingTerrain/paper/notes/extracted_text/_tmp_paper.txt)，论文方法主线可以概括为：

- 第 2 章：多特征贝叶斯风险概率图
- 第 3.2 节：稳健混合 LS-TLS
- 第 3.3 节：序列 PEIV / 抗差整体最小二乘
- 第 3.4 节：安全系数与安全着陆区搜索

### 6.2 当前代码与论文模块对照表

| 论文模块 | 当前对应代码 | 当前状态 | 说明 |
| --- | --- | --- | --- |
| 多特征风险概率图 | `code/terrain_features.py`、`code/risk_probability_map.py` | 已实现主干 | 已有局部方差、TPI、坡度均值、高程极差、多阈值、贝叶斯融合、协方差修正、MRF 风格平滑 |
| 稳健混合 LS-TLS | `code/robust_ls_tls.py` | 已实现实用版 | 已接入算法注册，但尚未进入当前权威全局结果表 |
| 序列 PEIV / 抗差 TLS | `code/sequential_robust_tls.py` | 已实现工程版 | 已具备分批更新、IGGIII 风格权重和二阶段修正，但仍偏工程近似版 |
| 安全系数计算 | `code/safety_evaluation.py` | 已实现 | 已有粗糙度、坡度、安全系数与窗口搜索 |
| 风险图 + 安全图联合决策 | `code/joint_risk_safety.py` | 已实现扩展版 | 这是当前仓库相对论文的一条工程增强线 |

### 6.3 当前已对齐的部分

当前已经基本对齐的部分有：

- 风险特征提取链
- 贝叶斯风险图主干
- 稳健混合 LS-TLS 的实用实现
- `RSTLS` 的工程化序列/分批版本
- 安全系数图与滑窗搜索
- 风险图与安全图联合决策

### 6.4 当前仍未完全对齐的部分

仍有 4 个差距需要明确：

1. 当前正式结果表还没有完整覆盖 `RobustLS-TLS`、风险图、安全图、联合图指标。
2. 当前 `RSTLS` 更接近“工程可用的分批稳健 TLS”，还不是论文公式级别的严格 PEIV 递推复现。
3. 当前正式结果主要还是随机离群点 sweep，不是更适合论文叙事的“结构化污染 + 安全区评价”。
4. 当前结果目录虽然已经拆成 `results/paper / results/debug / results/archive`，但里面的历史结果口径还需要继续收敛。

## 7. 当前最值得优先整理的事项

1. 统一结果目录，避免 `outlier_sweep_results` 多份并存。
2. 用当前新代码重新跑一版 `paper_structured35`，把 `RobustLS-TLS`、风险图、安全图、联合图正式写进结果树。
3. 重新生成 manifest，修正旧的 OneDrive 绝对路径。
4. 统一 CSV 列名。当前旧结果中仍出现 `Normal Deviation (бу)` 这类编码残留，而当前代码已经使用 `Normal Deviation (deg)`。
5. 继续把旧脚本限制在 `legacy/`，不要重新混回主入口。

## 8. 当前建议的下一步

如果目标是“把项目整理成一份可以继续写论文和继续同步 GitHub 的研究仓库”，最推荐的顺序是：

1. 保持当前新目录结构不再回退。
2. 先以 `structured35` 重跑正式结果。
3. 让 `RobustLS-TLS`、`RSTLS`、风险图、安全图和联合图都进入统一结果树。
4. 再基于新结果补正文图和论文结论。

一句话总结当前状态：

> 当前仓库的代码主线已经基本成型，论文方法链的大部分关键模块也已经有实现；真正欠缺的不是“再写一套新代码”，而是把现有代码跑出一套统一、正式、可引用的结构化结果，并把旧结果树继续收拢。
