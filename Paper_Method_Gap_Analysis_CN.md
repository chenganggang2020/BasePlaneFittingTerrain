# 论文方法与当前代码对照分析

对应文档：

- `D:\OneDrive\Zotero\storage\避障着陆\基准面拟合\基于激光雷达的深空软着陆稳健障碍检测方法研究.pdf`
- 提取文本：`D:\OneDrive\python\BasePlaneFittingTerrain\_tmp_paper.txt`

## 0. 实现进展（2026-04-04）

相对于本文档最初编写时的状态，当前仓库已经补上了以下模块：

- `safety_evaluation.py`
  - 已实现论文第 3.4 节对应的粗糙度、局部坡度、安全系数矩阵、滑窗安全区搜索与可视化。
- `robust_ls_tls.py`
  - 已实现论文第 3.2 节对应的稳健混合 LS-TLS 实用版，并接入 `algorithm_runner.py`。
- `sequential_robust_tls.py`
  - 已重写为分组递推的信息阵更新版本，并加入观测空间/结构空间的双因子 IGGIII 风格稳健权。
- `terrain_features.py`
  - 已实现论文第 2 章风险概率图所需的局部方差、TPI、局部坡度均值、高程极差提取。
- `risk_probability_map.py`
  - 已实现多阈值联合判定、线性加权风险图、贝叶斯概率融合、特征相关性修正、MRF 风格平滑近似。
- `joint_risk_safety.py`
  - 已实现风险概率图与安全系数图的联合评价、联合危险图/安全得分图以及联合窗口搜索。
- `monte_carlo.py`
  - 已接入风险概率图分析、安全系数分析以及联合风险-安全评价的结果导出。
- `experiment_presets.py` / `run_paper_experiments.py` / `run_ablation_study.py`
  - 已补论文主实验入口与消融实验入口，并支持方法筛选和模块开关。

因此，下面的“差距分析”应理解为：

- 一部分仍然是真实缺口，例如“严格论文版序列 PEIV 递推”。
- 另一部分已经从“缺失”变成了“已有实用实现，但仍可继续向论文细节逼近”。

## 1. 结论摘要

当前仓库已经具备以下基础：

- 模拟点云/DEM 生成
- 多种平面拟合算法的统一实验入口
- 批量扫描不同离群点比例的实验框架
- 结果导出与可视化

但从论文复现角度看，当前实现仍属于“实验平台 + 部分算法原型”，并未完整实现论文的核心方法链。

最关键的差距有 4 类：

1. 第 2 章“风险概率图”方法基本未实现。
2. 第 3.2 节“稳健混合 LS-TLS”未实现为独立算法。
3. 第 3.3 节“序列抗差整体最小二乘”仅有近似版，未实现真正的分组递推/序列 PEIV。
4. 第 3.4 节“安全系数 + 子窗口安全区搜索”未实现。

## 2. 论文关键方法拆解

根据 `_tmp_paper.txt`，论文主要方法链包括：

### 2.1 大范围风险概率图

来源：

- `_tmp_paper.txt` 第 2 章
- 特征：局部方差、TPI、局部坡度均值、高程极差
- 融合：多阈值、线性加权、多特征贝叶斯概率融合
- 修正：协方差矩阵修正、MRF 后处理优化

方法模块：

- DEM 地形特征计算
- 多特征归一化/概率化
- 贝叶斯融合
- 特征相关性修正
- MRF 空间连续性优化

### 2.2 小范围基准面拟合与障碍提取

来源：

- `_tmp_paper.txt` 第 3 章

方法模块：

- 基线算法：LS / WTLS / RWTLS / RANSAC / LMedsq / PDIMSE
- 论文方法 1：稳健混合 LS-TLS
- 论文方法 2：稳健序列 PEIV 抗差整体最小二乘
- 基于拟合平面的粗糙度、局部坡度、安全系数计算
- 子窗口安全区搜索

## 3. 当前代码模块对照

### 3.1 已有模块

- `OutlierTerrainSimulator.py`
  - 作用：生成带离群点的模拟 DEM 和点云。
- `monte_carlo.py`
  - 作用：批量读取点云、运行算法、汇总结果、出图。
- `algorithm_runner.py`
  - 作用：注册算法列表。
- `ls.py`
  - 已实现：LS。
- `wtls.py`
  - 已实现：WTLS 原型。
- `rwtls.py`
  - 已实现：RWTLS 原型。
- `ransac.py`
  - 已实现：RANSAC。
- `lmedsq.py`
  - 已实现：LMedsq。
- `pdimse.py`
  - 已实现：PDIMSE。
- `ikose.py`
  - 已实现：IKOSE 扩展方法。
- `sequential_robust_tls.py`
  - 已实现：一个带 RANSAC 初始化 + IGG3 权重的 PEIV/TLS 拟合器原型。
- `visualizer.py`
  - 已实现：实验图件绘制。

### 3.2 论文方法与现代码状态表

| 论文模块 | 当前状态 | 判断 |
| --- | --- | --- |
| LS | 已实现 | 基本齐全 |
| WTLS | 已实现但未纳入主实验 | 部分完成 |
| RWTLS | 已实现原型 | 部分完成 |
| RANSAC / LMedsq / PDIMSE | 已实现 | 可用于对照实验 |
| 稳健混合 LS-TLS | 缺失 | 未实现 |
| 序列 PEIV 抗差整体最小二乘 | 近似实现 | 方法不完全一致 |
| 粗糙度计算 | 近似实现 | 当前多用“点到平面正交距离”代替论文定义 |
| 局部坡度计算 | 缺失 | 未见论文式实现 |
| 安全系数矩阵 C | 缺失 | 未实现 |
| 子窗口安全着陆区搜索 | 缺失 | 未实现 |
| 局部方差 / TPI / 局部坡度均值 / 高程极差 | 缺失 | 未实现 |
| 贝叶斯风险概率图 | 缺失 | 未实现 |
| 协方差修正 | 缺失 | 未实现 |
| MRF 后处理 | 缺失 | 未实现 |

## 4. 具体差距分析

### 4.1 风险概率图部分：基本缺失

论文要求：

- 计算局部方差、TPI、局部坡度均值、高程极差。
- 形成多阈值联合判定、线性加权图、贝叶斯风险概率图。
- 使用协方差修正与 MRF 优化。

当前仓库情况：

- 没有发现与 `Bayes`、`MRF`、`TPI`、`安全系数`、`风险概率图` 对应的实现。
- 也没有 DEM 特征提取模块。

结论：

- 第 2 章目前没有真正进入代码实现阶段。

### 4.2 稳健混合 LS-TLS：未实现

论文要求：

- 混合 LS-TLS。
- RANSAC 预筛选。
- MAD 自适应尺度估计。
- Huber 权函数。
- 稀疏化/增量权重更新。

当前仓库情况：

- `wtls.py` 和 `rwtls.py` 是 WTLS/RWTLS 原型。
- 但没有独立的 `robust_ls_tls.py` 或等价实现。
- `algorithm_runner.py` 中也没有“稳健 LS-TLS”算法入口。

结论：

- 论文第 3.2 节关键创新点未实现。

### 4.3 序列抗差 PEIV：只实现了“稳健 PEIV 拟合器原型”，未实现真正序列法

论文要求：

- PEIV 模型。
- 分组递推、序列平差。
- IGGIII 双因子抗差。
- 在观测空间与结构空间保持相关性。

当前仓库情况：

- `sequential_robust_tls.py` 名称接近论文方法。
- 但实现仍是对整批点云一次性中心化后迭代求解，不是“按组递推继承历史信息”的序列平差。
- 也没有明确的 group/block 输入或递推状态对象。

结论：

- 当前实现适合作为“序列法的近似起点”，但不能称为论文第 3.3 节的完整复现。

### 4.4 当前“粗糙度”定义与论文不完全一致

论文定义：

- 粗糙度 `R(u,v) = H(u,v) - Hs(u,v)`，即高程相对基准面的垂向差值。
- 局部坡度由地形法向量与基准面法向量夹角给出。

当前代码情况：

- `monte_carlo.py` 统一将残差定义为点到平面的正交距离。
- `visualizer.py` 直接把绝对残差图命名为 roughness。

影响：

- 实验图看起来合理，但与论文指标口径不一致。
- 后续若要复现实验表格或安全系数图，会出现定义错位。

### 4.5 安全系数与安全区搜索：未实现

论文要求：

- 若 `R >= h_threshold` 或 `theta >= theta_threshold`，则 `C=0`。
- 否则按公式计算安全系数 `C(u,v)`。
- 按着陆器尺寸滑动窗口搜索。
- 若窗口 `min(C)=0`，则窗口危险。
- 否则取窗口均值最大的区域作为最优安全着陆区。

当前仓库情况：

- 目前没有 `safety_evaluation.py` 或等价模块。
- `plot_roughness_heatmap()` 只是可视化，不是安全区搜索。

结论：

- 论文第 3.4 节尚未落地。

## 5. 当前代码中值得优先修正的问题

### 5.1 研究口径问题

- 现有实验指标偏向“平面拟合结果对比”，还没有进入“障碍检测效果对比”。
- 论文真正想比较的是：
  - 平面参数误差
  - 安全系数图误差
  - 障碍检测/安全着陆区选择效果

### 5.2 方法命名与实现不完全一致

- `SequentialRobustTLSPlaneFitter` 的命名容易让人误以为已完成“序列递推 PEIV”。
- 实际更接近“稳健 PEIV/TLS 迭代拟合器”。

建议：

- 在真正补完前，把方法命名改得更精确，避免论文复现时概念混淆。

### 5.3 指标定义混用

- 当前“roughness”图多是正交残差。
- 论文“粗糙度”是相对基准面的高程差。

建议：

- 把 `orthogonal_residual`、`vertical_residual`、`roughness` 三者分开。

### 5.4 缺少 DEM-点云双通路

- 第 2 章依赖 DEM 栅格特征。
- 第 3 章依赖点云平面拟合。

当前仓库主要是点云实验框架，缺少 DEM 特征分析总线。

## 6. 推荐的补齐顺序

建议按“先忠实复现，再做创新”两阶段推进。

### 阶段 A：忠实复现论文核心链路

#### A1. 先补基础指标与安全区模块

新增建议：

- `terrain_features.py`
  - `compute_weighted_local_variance(dem, window_size, sigma)`
  - `compute_tpi(dem, window_size)`
  - `compute_local_mean_slope(dem, window_size, method='sobel')`
  - `compute_elevation_range(dem, window_size)`
- `safety_evaluation.py`
  - `compute_vertical_roughness(dem, plane_params)`
  - `compute_local_slope(dem, plane_params)`
  - `compute_safety_coefficient(roughness, slope, h_threshold, theta_threshold)`
  - `search_safe_windows(C, window_size)`

原因：

- 这部分依赖最少。
- 也是把论文从“平面拟合实验”推进到“障碍检测应用”的关键一步。

#### A2. 实现稳健混合 LS-TLS

新增建议：

- `robust_ls_tls.py`

至少包含：

- RANSAC 预筛
- 混合 LS-TLS 主体
- MAD 尺度估计
- Huber 权重
- 自适应阈值
- 稀疏/增量权重更新

同时：

- 在 `algorithm_runner.py` 中注册为 `Robust_LS_TLS` 或 `RobustLS-TLS`

#### A3. 把序列 PEIV 做成真正递推版

新增或重构建议：

- `sequential_peiv_tls.py`

建议能力：

- 支持 `fit_blocks(blocks)` 或 `update(block)` 的 API
- 显式保存递推状态
- 实现组间信息继承
- 实现观测空间/结构空间双因子 IGGIII

建议：

- 现有 `sequential_robust_tls.py` 可作为数值参考，但不建议直接继续堆逻辑。
- 更稳妥的做法是保留它，另写真正的序列版。

#### A4. 补第 2 章风险概率图

新增建议：

- `risk_probability_map.py`

模块内容：

- 特征归一化/概率化
- 多阈值联合判定
- 线性加权融合
- 贝叶斯融合
- 协方差修正
- MRF 后处理

### 阶段 B：做出更强创新

在论文复现完成之后，优先做下面 3 条。

#### B1. 分层障碍检测

思路：

- 大尺度区域：DEM 风险概率图
- 小尺度区域：局部稳健参考面 + 安全系数
- 最终：概率风险与几何安全联合决策

价值：

- 直接解决论文中“风险图与安全系数图各自有盲点”的问题。

#### B2. 概率安全系数

思路：

- 不是只计算一个确定性 `C(u,v)`。
- 把平面参数不确定度、测量噪声、不确定阈值传播进去。
- 输出 `P(safe)` 或 `P(roughness<h, slope<theta)`。

价值：

- 创新性明显强于只换一个权函数。
- 也更接近真实着陆决策。

#### B3. 自适应参考地形模型

思路：

- 平面只是局部近似。
- 对于缓坡、鞍部、轻微曲面，可以采用：
  - 分片平面
  - 二次曲面
  - B-spline 曲面
  - 模型选择或模型融合

价值：

- 直接针对论文中“在斜坡区域出现低风险、低安全矛盾”的问题。

## 7. 建议的目录重构

当前文件较分散，建议逐步收敛为：

- `sim/`
  - `terrain_simulator.py`
- `algorithms/`
  - `ls.py`
  - `wtls.py`
  - `rwtls.py`
  - `robust_ls_tls.py`
  - `sequential_peiv_tls.py`
- `hazard/`
  - `terrain_features.py`
  - `risk_probability_map.py`
  - `safety_evaluation.py`
- `experiments/`
  - `run_outlier_sweep.py`
  - `run_risk_map_demo.py`
  - `run_safe_landing_demo.py`
- `viz/`
  - `visualizer.py`

如果暂时不想大改结构，也至少建议先新增：

- `robust_ls_tls.py`
- `terrain_features.py`
- `risk_probability_map.py`
- `safety_evaluation.py`

## 8. 最推荐的下一步

如果目标是“尽快把论文做成可复现实验”，建议下一步按下面顺序做：

1. 先实现 `safety_evaluation.py`
2. 再实现 `robust_ls_tls.py`
3. 再重写真正的 `sequential_peiv_tls.py`
4. 最后补 `risk_probability_map.py`

原因：

- 第 3 章更接近你现在已有代码，接入成本最低。
- 第 2 章风险图需要新建 DEM 特征管线，工作量更大。

## 9. 我建议的实施策略

推荐分三轮完成：

### 第一轮

- 补 `safety_evaluation.py`
- 把“粗糙度/坡度/安全系数/窗口搜索”跑通
- 让当前已有算法可以按论文口径出安全区结果

### 第二轮

- 增加 `robust_ls_tls.py`
- 把它接入 `algorithm_runner.py`
- 新增按论文口径的基准面拟合比较实验

### 第三轮

- 重构序列 PEIV 为真正递推版
- 补风险概率图
- 做“风险图 + 安全系数图”的联合对比实验

## 10. 当前最关键的判断

如果只问一句“现在代码实现论文了吗”，答案是：

- 没有完整实现。
- 已实现的是：若干基线算法 + 仿真平台 + 部分稳健拟合原型。
- 论文最有代表性的创新模块目前还缺 3 大块：
  - 稳健混合 LS-TLS
  - 风险概率图
  - 安全系数与安全区搜索

如果只问一句“接下来最值得先写什么”，答案是：

- 先写 `safety_evaluation.py` 和 `robust_ls_tls.py`。
