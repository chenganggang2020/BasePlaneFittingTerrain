import os

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None
from scipy.interpolate import griddata
from scipy.spatial.distance import cosine
import time
from matplotlib.gridspec import GridSpec
from constants import ALGORITHM_COLORS, ALGORITHM_MARKERS  # Ensure import of style constants
from matplotlib.patches import Patch

# Global configuration (remove Helvetica, ensure English compatibility)
plt.rcParams["font.family"] = ["Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.max_open_warning"] = 50  # Increase number of open figures allowed
plt.rcParams['agg.path.chunksize'] = 1000  # Resolve Agg rendering path size issue
plt.rcParams['path.simplify_threshold'] = 0.5  # Simplify paths to reduce point count
_cache = {}  # For caching intermediate results (shared preprocessing for 3D terrain/outliers)


# ---------------------- 1. General Downsampling Function (shared across all modules) ----------------------
def downsample_points(points, ratio=0.1, random_state=42):
    """Intelligent downsampling: returns downsampled points and their original indices (solves index matching issue)"""
    n_points = len(points)
    if n_points <= 100000:  # No sampling for small point clouds (avoid detail loss)
        return points.copy(), np.arange(n_points)

    # For very large point clouds, keep at most 100,000 points (balance visualization and performance)
    max_keep = 100000
    ratio = min(ratio, max_keep / n_points)
    rng = np.random.default_rng(random_state)
    indices = rng.choice(n_points, int(n_points * ratio), replace=False)
    return points[indices], indices


# ---------------------- 2. 3D Terrain/Outlier Specialized Preprocessing (shared cache) ----------------------
def preprocess_3d_data(points,labels=None):
    """Preprocess data for 3D terrain/outlier analysis, cache downsampling indices and original coordinates"""
    cached_points = _cache.get('3d_points')
    cached_labels = _cache.get('labels_source')
    labels_match = (
        (labels is None and cached_labels is None) or
        (labels is not None and cached_labels is not None and np.array_equal(cached_labels, labels))
    )
    if cached_points is not None and labels_match and np.array_equal(cached_points, points):
        return _cache  # Already cached, return directly

    # Convert to float32 to reduce memory usage
    points = points.astype(np.float32)

    # Downsample point cloud (for visualization) + save original indices (critical: ensure mask matching later)
    downsampled, down_indices = downsample_points(points, ratio=0.05)  # Keep 100,000 out of 2,000,000

    # Extract coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    x_ds, y_ds, z_ds = downsampled[:, 0], downsampled[:, 1], downsampled[:, 2]

    # Precompute X-direction group means (for elevation trend plot)
    x_unique, x_idx = np.unique(x, return_inverse=True)
    sum_z = np.bincount(x_idx, weights=z)
    count_z = np.bincount(x_idx)
    z_mean = sum_z / count_z  # Vectorized calculation, avoid loops

    # Update cache (explicit key names to avoid conflicts)
    _cache.update({
        '3d_points': points,
        'downsampled': downsampled,
        'down_indices': down_indices,  # Indices of downsampled points in original cloud
        'x': x, 'y': y, 'z': z,
        'x_ds': x_ds, 'y_ds': y_ds, 'z_ds': z_ds,
        'x_unique': x_unique,
        'z_mean': z_mean,
        'labels_ds': labels[down_indices] if labels is not None and len(labels) == len(points) else labels,
        'labels_source': None if labels is None else np.asarray(labels).copy(),
    })
    return _cache


def coerce_inlier_mask(inliers, point_count, method_name=None):
    if point_count <= 0:
        return np.array([], dtype=bool)
    if inliers is None:
        return np.ones(point_count, dtype=bool)

    arr = np.asarray(inliers)
    if arr.size == 0:
        return np.zeros(point_count, dtype=bool)

    label = f" for {method_name}" if method_name else ""
    if np.issubdtype(arr.dtype, np.bool_):
        if arr.size != point_count:
            print(f"Warning: inliers mask length mismatch{label}; using all points for visualization.")
            return np.ones(point_count, dtype=bool)
        return arr.astype(bool, copy=False)

    if np.issubdtype(arr.dtype, np.integer):
        flat = arr.reshape(-1)
        if flat.size == point_count and np.all(np.isin(flat, [0, 1])):
            return flat.astype(bool)
        mask = np.zeros(point_count, dtype=bool)
        valid = flat[(flat >= 0) & (flat < point_count)]
        if valid.size != flat.size:
            print(f"Warning: invalid inlier indices detected{label}; clipping to valid range.")
        mask[valid] = True
        return mask

    print(f"Warning: unsupported inliers format{label}; using all points for visualization.")
    return np.ones(point_count, dtype=bool)


# ---------------------- 3. 3D Terrain Plotting (fixed cache calls, optimized rendering) ----------------------


def plot_common_3d_terrain(points, fitting_results=None, labels=None,
                           show_plot=False, save_plot=True, save_path=None, dpi=300):
    """
    紧凑布局的对比图：
    - 左侧原始点云，右侧算法子图
    - 无格网、无坐标轴，仅显示点云和拟合平面
    - 图例放在整个图上方，布局紧凑
    - 支持不传入labels：此时不区分内外点，统一绘制原始点云
    """
    start_time = time.time()
    # 预处理数据（支持labels为None）
    cache = preprocess_3d_data(points, labels)
    print(f"3D data preprocessing time: {time.time() - start_time:.2f}s")

    # 处理算法结果默认值
    if fitting_results is None:
        fitting_results = []
    n_algorithms = len(fitting_results)
    if n_algorithms == 0:
        raise ValueError("fitting_results cannot be empty")

    # 检查labels长度（仅当labels存在时）
    if labels is not None and len(labels) != len(points):
        raise ValueError("labels length must match points length")

    # --------------------------
    # 布局参数（极致紧凑）
    # --------------------------
    max_cols = min(n_algorithms, 4)  # 最多4列（避免过宽）
    n_rows = (n_algorithms + max_cols - 1) // max_cols  # 计算行数

    # 调整图片尺寸：极致紧凑
    fig_width = 0 + max_cols * 4  # 每列更窄
    fig_height = 0 + n_rows * 4   # 每行更矮（关键：大幅降低高度）

    # 设置统一字体大小
    plt.rcParams.update({
        'font.size': 6,
        'axes.labelsize': 6,
        'axes.titlesize': 7,
        'legend.fontsize': 6,
    })

    fig = plt.figure(figsize=(fig_width, fig_height))

    # 添加整体标题
    fig.suptitle("Point Cloud Plane Fitting Results", fontsize=9, y=0.94)  # 标题位置更紧凑

    # 网格布局（极致紧凑）
    gs = GridSpec(n_rows, max_cols + 1,
                  width_ratios=[1.0] + [0.6] * max_cols,  # 右侧更窄
                  wspace=0.08, hspace=0.08)  # 极致紧凑的间距

    # --------------------------
    # 左侧子图：原始点云（无格网、无坐标）
    # --------------------------
    ax1 = fig.add_subplot(gs[:, 0], projection='3d')
    ax1.view_init(elev=30, azim=45)

    # 关闭格网和坐标轴
    ax1.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.set_zlabel('')
    ax1.set_axis_off()

    # 绘制原始点云（支持无labels的情况）
    if labels is not None:
        # 有labels：区分内外点
        inliers_original = cache['labels_ds'] == 1
        outliers_original = cache['labels_ds'] == 0
        ax1.scatter(
            cache['x_ds'][inliers_original], cache['y_ds'][inliers_original], cache['z_ds'][inliers_original],
            c='blue', s=0.6, alpha=0.8,
            label='Inliers', zorder=1
        )
        ax1.scatter(
            cache['x_ds'][outliers_original], cache['y_ds'][outliers_original], cache['z_ds'][outliers_original],
            c='red', s=0.6, alpha=0.8,
            label='Outliers', zorder=1
        )
    else:
        # 无labels：不区分内外点，统一颜色
        ax1.scatter(
            cache['x_ds'], cache['y_ds'], cache['z_ds'],
            c='blue', s=0.6, alpha=0.8,
            label='Points', zorder=1
        )

    # 标题
    ax1.set_title('Original', fontsize=7, pad=2)

    # --------------------------
    # 右侧子图：算法结果（无格网、无坐标）
    # --------------------------
    x_grid = np.linspace(cache['x'].min(), cache['x'].max(), 50)
    y_grid = np.linspace(cache['y'].min(), cache['y'].max(), 50)
    X, Y = np.meshgrid(x_grid, y_grid)
    display_points = cache['downsampled']
    display_indices = cache['down_indices']

    for i, fr in enumerate(fitting_results):
        row = i // max_cols
        col = i % max_cols + 1  # 右侧列索引（从1开始）
        ax = fig.add_subplot(gs[row, col], projection='3d')
        ax.view_init(elev=30, azim=45)  # 统一视角
        ax.set_axis_off()

        # 关闭格网和坐标轴
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')

        # 算法信息
        method_name = fr["method"]
        n = fr["n_est"]
        d = fr["d_est"]
        inliers = coerce_inlier_mask(fr.get("inliers"), len(points), method_name=method_name)
        display_inliers = inliers[display_indices]

        # 绘制点云（算法识别的内外点）
        ax.scatter(
            display_points[display_inliers, 0], display_points[display_inliers, 1], display_points[display_inliers, 2],
            c='blue', s=0.6, alpha=0.8,
            label='Inliers', zorder=2
        )
        ax.scatter(
            display_points[~display_inliers, 0], display_points[~display_inliers, 1], display_points[~display_inliers, 2],
            c='red', s=0.6, alpha=0.8,
            label='Outliers', zorder=2
        )

        # 绘制拟合平面
        if np.isclose(n[2], 0):
            Z = np.zeros_like(X) + (-(n[0] * X + n[1] * Y + d) / (n[2] + 1e-8))
        else:
            Z = -(n[0] * X + n[1] * Y + d) / n[2]
        # 假设ALGORITHM_COLORS已定义，若未定义可替换为默认颜色
        plane_color = ALGORITHM_COLORS.get(method_name, "#aaaaaa")
        ax.plot_surface(
            X, Y, Z,
            color=plane_color,
            alpha=0.4,
            linewidth=0,  # 去掉平面网格线
            zorder=3
        )

        # 标题
        ax.set_title(method_name, fontsize=7, pad=2)

    # --------------------------
    # 图例放在整个图上方（关键修改）
    # --------------------------
    # 创建一个假的图例用于显示（不实际绘制任何内容）
    import matplotlib.patches as mpatches
    inlier_patch = mpatches.Patch(color='blue', label='Inliers')
    outlier_patch = mpatches.Patch(color='red', label='Outliers')

    fig.legend(handles=[inlier_patch, outlier_patch],
               loc='upper center',
               bbox_to_anchor=(0.5, 0.90),  # 位置在图上方
               fontsize=6,
               ncol=2)  # 两列并排

    # 保存时强制裁剪空白区域（最紧凑）
    if save_plot and save_path:
        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches='tight',  # 裁剪多余空白
            pad_inches=0.02,  # 最小边距
            facecolor='white',  # 背景色
            edgecolor='none'  # 无边框
        )
        print(f"Comparison plot saved to: {save_path}")

    if show_plot:
        plt.show()

    print(f"Total plotting time: {time.time() - start_time:.2f}s")
    return fig

# ---------------------- 4. Outlier Distribution Plotting (fixed index matching, ensure mask length consistency) ----------------------
def plot_common_outlier_dist(points, inliers, x_label, y_label,
                             show_plot=True, save_plot=True, save_path=None):
    """Optimized outlier distribution: use downsampling indices to avoid mask length mismatch errors"""
    start_time = time.time()
    cache = preprocess_3d_data(points)

    # Critical fix: filter inliers by downsampling indices (ensure matching downsampled point count)
    inliers = coerce_inlier_mask(inliers, len(cache['3d_points']))
    inliers_ds = inliers[cache['down_indices']]  # Inliers mask for downsampled points

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    # Plot downsampled inliers/outliers (color differentiation, semi-transparent to avoid overlap)
    outlier_mask = ~inliers_ds
    ax.scatter(
        cache['x_ds'][inliers_ds], cache['y_ds'][inliers_ds],
        c='skyblue', s=1, alpha=0.6, label='Inliers (Baseline)'
    )
    ax.scatter(
        cache['x_ds'][outlier_mask], cache['y_ds'][outlier_mask],
        c='crimson', s=1, alpha=0.8, label='Outliers (Baseline)'
    )

    # Label and title optimization
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title('Common Outlier Distribution (Baseline: LS Method)', fontsize=14, pad=15)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    # 保存图像
    if save_plot:
        if not save_path:
            print("Warning: save_plot=True but save_path is None, skipping save")
        else:
            fig.savefig(save_path, dpi=1200, bbox_inches='tight')
            print(f"Outlier distribution plot saved to: {save_path}")

    # 显示图像
    if show_plot:
        plt.show()

    print(f"Outlier distribution plotting time: {time.time() - start_time:.2f}s")
    return fig


# ---------------------- 5. Roughness Heatmap (optimized interpolation speed, reduced memory usage) ----------------------
def plot_roughness_heatmap(
        points, residuals, method_name,
        downsample_max=50000, grid_size=100,
        show_plot=True, save_plot=True, save_path=None
):
    """
    绘制单个算法的粗糙度热图（可单独调用）

    参数：
        points: 点云坐标 (N, 3)
        residuals: 残差值 (N,)
        method_name: 算法名称（用于标题）
        downsample_max: 最大插值点数（控制速度）
        grid_size: 插值网格边长（控制精度）
        show_plot: 是否显示图像（默认True）
        save_plot: 是否保存图像（默认True）
        save_path: 保存路径（仅save_plot=True时生效，如"heatmap.png"）
    返回：matplotlib的fig对象
    """
    start_time = time.time()
    # 下采样
    interp_ratio = min(downsample_max / len(points), 1.0)
    interp_points, interp_indices = downsample_points(points, ratio=interp_ratio)
    interp_residuals = residuals[interp_indices]

    # 坐标范围与插值网格
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(xi, yi)
    residual_grid = griddata(
        points=(interp_points[:, 0], interp_points[:, 1]),
        values=interp_residuals,
        xi=(X, Y),
        method='linear',
        fill_value=0
    )

    # 绘图
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    heatmap = ax.imshow(
        residual_grid,
        extent=[x_min, x_max, y_min, y_max],
        cmap='RdYlGn_r',
        aspect='auto',
        origin='lower'
    )
    plt.colorbar(heatmap, ax=ax, label='Roughness (m) - Absolute Residual')

    # 标签设置
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'Roughness Heatmap ({method_name})', fontsize=14, pad=15)
    plt.tight_layout()

    # 保存图像
    if save_plot:
        if not save_path:
            print("Warning: save_plot=True but save_path is None, skipping save")
        else:
            fig.savefig(save_path, dpi=1200, bbox_inches='tight')
            print(f"[{method_name}] Heatmap saved to: {save_path}")

    # 显示图像
    if show_plot:
        plt.show()

    print(f"[{method_name}] Roughness heatmap time: {time.time() - start_time:.2f}s")
    return fig


# ---------------------- 6. Elevation Profile Specialized Preprocessing (renamed to avoid overwriting) ----------------------
def preprocess_elevation_bins(points, bin_width=1.0):
    """高程数据分箱预处理（供所有高程图函数复用）"""
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    x_min, x_max = np.min(x), np.max(x)
    bins = np.arange(x_min, x_max + bin_width, bin_width)
    x_bin_centers = (bins[:-1] + bins[1:]) / 2

    bin_indices = np.digitize(x, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

    z_median = []
    y_mean_per_bin = []
    for i in range(len(bins) - 1):
        mask = (bin_indices == i)
        if np.any(mask):
            z_median.append(np.median(z[mask]))
            y_mean_per_bin.append(np.mean(y[mask]))
        else:
            z_median.append(np.nan)
            y_mean_per_bin.append(np.nan)
    z_median = np.array(z_median)
    y_mean_per_bin = np.array(y_mean_per_bin)

    valid_idx_z = np.where(~np.isnan(z_median))[0]
    z_median_filled = np.interp(np.arange(len(z_median)), valid_idx_z, z_median[valid_idx_z])

    valid_idx_y = np.where(~np.isnan(y_mean_per_bin))[0]
    y_mean_filled = np.interp(np.arange(len(y_mean_per_bin)), valid_idx_y, y_mean_per_bin[valid_idx_y])

    return {
        'x': x, 'y': y, 'z': z,
        'x_bin_centers': x_bin_centers,
        'z_median_filled': z_median_filled,
        'y_mean_filled': y_mean_filled
    }


# ---------------------- 7. Elevation Profile Plotting (50% transparency, optimized trend display) ----------------------
def plot_elevation_profile(
        points, n_est, d_est, method_name, bin_width=1.0,
        show_plot=True, save_plot=True, save_path=None
):
    """
    绘制单个算法的高程剖面图（可单独调用）

    参数：
        points: 点云坐标 (N, 3)
        n_est: 拟合平面法向量 (3,)
        d_est: 拟合平面参数 (标量)
        method_name: 算法名称（用于标题/图例）
        bin_width: X轴分箱宽度
        show_plot: 是否显示图像（默认True）
        save_plot: 是否保存图像（默认True）
        save_path: 保存路径（仅save_plot=True时生效，如"elevation.png"）
    返回：matplotlib的fig对象
    """
    start_time = time.time()
    # 预处理与拟合计算
    cache = preprocess_elevation_bins(points, bin_width=bin_width)
    y_mean = np.mean(points[:, 1])
    z_plane_per_bin = -(n_est[0] * cache['x_bin_centers'] + n_est[1] * y_mean + d_est) / n_est[2]

    # 绘图
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    # 原始数据与拟合线
    ax.plot(
        cache['x_bin_centers'], cache['z_median_filled'],
        'b-', linewidth=1.5, alpha=0.5,
        label='Original Data (Median Trend)'
    )
    ax.plot(
        cache['x_bin_centers'], z_plane_per_bin,
        'r-', linewidth=2, alpha=0.5,
        label=f'Fitted Plane ({method_name})'
    )

    # 参考线
    ax.axhline(y=np.median(points[:, 2]), color='gray', linestyle=':', alpha=0.5,
               label='Original Global Median')
    ax.axhline(y=np.mean(z_plane_per_bin), color='orange', linestyle=':', alpha=0.5,
               label=f'Plane Global Mean ({method_name})')

    # 标签设置
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Elevation (m)', fontsize=12)
    ax.set_title(f'Elevation Profile ({method_name})', fontsize=14, pad=15)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    # 保存图像
    if save_plot:
        if not save_path:
            print("Warning: save_plot=True but save_path is None, skipping save")
        else:
            fig.savefig(save_path, dpi=1200, bbox_inches='tight')
            print(f"[{method_name}] Elevation profile saved to: {save_path}")

    # 显示图像
    if show_plot:
        plt.show()

    print(f"[{method_name}] Elevation profile time: {time.time() - start_time:.2f}s")
    return fig


def plot_elevation_trend_comparison(
        points, fitting_results, bin_width=1.0,
        show_plot=True, save_plot=True, save_path=None
):
    """
    多算法高程趋势对比图（使用统一配色）
    """
    start_time = time.time()
    # 过滤有效算法
    valid_results = [
        fr for fr in fitting_results
        if "n_est" in fr and "d_est" in fr and "method" in fr
           and not np.isnan(fr["n_est"]).any() and not np.isnan(fr["d_est"])
    ]
    if not valid_results:
        print("Warning: no valid results for elevation trend comparison")
        return None

    # 预处理与计算
    cache = preprocess_elevation_bins(points, bin_width=bin_width)  # 假设该函数已定义
    y_mean = np.mean(points[:, 1])

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 原始数据趋势线
    ax.plot(
        cache['x_bin_centers'], cache['z_median_filled'],
        'b-', linewidth=1.5, alpha=0.5,
        label='Original Data (Median Trend)',
        zorder=10
    )

    # 多算法拟合线（使用ALGORITHM_COLORS配色，关键修改）
    for fr in valid_results:
        method_name = fr["method"]  # 获取算法名称
        n, d = fr["n_est"], fr["d_est"]
        # 计算拟合平面的高程趋势
        z_plane_per_bin = -(n[0] * cache['x_bin_centers'] + n[1] * y_mean + d) / n[2]

        # 从预设配色中获取颜色，未定义的算法用默认灰色
        line_color = ALGORITHM_COLORS.get(method_name, "#aaaaaa")

        ax.plot(
            cache['x_bin_centers'], z_plane_per_bin,
            color=line_color,  # 使用预设颜色（核心修改）
            linestyle='--', linewidth=2, alpha=0.8,  # 适当提高透明度区分线条
            label=method_name,
            zorder=5
        )

    # 参考线与标签
    ax.axhline(y=np.median(points[:, 2]), color='gray', linestyle=':', alpha=0.5,
               label='Original Global Median')
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Elevation (m)', fontsize=12)
    ax.set_title('Elevation Trend: Original vs. Fitted Planes', fontsize=14, pad=15)
    ax.legend(loc='best', fontsize=10)
    ax.grid(linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()

    # 保存图像
    if save_plot:
        if not save_path:
            print("Warning: save_plot=True but save_path is None, skipping save")
        else:
            fig.savefig(save_path, dpi=1200, bbox_inches='tight')
            print(f"Elevation trend comparison saved to: {save_path}")

    # 显示图像
    if show_plot:
        plt.show()

    print(f"Elevation trend comparison time: {time.time() - start_time:.2f}s")
    return fig


# ---------------------- 8. Elevation Trend Specialized Preprocessing (unified key names, avoid KeyError) ----------------------
def preprocess_elevation_trend(points, bin_width=1.0):
    """
    Specialized preprocessing for elevation trend: consistent logic with elevation profile, unified key names
    Fixed: original bins_bin_centers→x_bin_centers key name error
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]  # Z-axis is elevation

    # Equal-width binning
    x_min, x_max = np.min(x), np.max(x)
    bins = np.arange(x_min, x_max + bin_width, bin_width)
    x_bin_centers = (bins[:-1] + bins[1:]) / 2  # Unified key name: x_bin_centers

    # Bin index (prevent out-of-bounds)
    bin_indices = np.digitize(x, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

    # Calculate bin median (original data) and Y mean (for fitted plane)
    z_median = []
    y_mean_per_bin = []
    for i in range(len(bins) - 1):
        mask = (bin_indices == i)
        if np.any(mask):
            z_median.append(np.median(z[mask]))
            y_mean_per_bin.append(np.mean(y[mask]))
        else:
            z_median.append(np.nan)
            y_mean_per_bin.append(np.nan)
    z_median = np.array(z_median)
    y_mean_per_bin = np.array(y_mean_per_bin)

    # Interpolation to fill empty bins (avoid broken trend lines)
    valid_idx_z = np.where(~np.isnan(z_median))[0]
    z_median_filled = np.interp(np.arange(len(z_median)), valid_idx_z, z_median[valid_idx_z])

    valid_idx_y = np.where(~np.isnan(y_mean_per_bin))[0]
    y_mean_filled = np.interp(np.arange(len(y_mean_per_bin)), valid_idx_y, y_mean_per_bin[valid_idx_y])

    return {
        'x': x,
        'y': y,
        'z': z,
        'x_bin_centers': x_bin_centers,  # Fixed key name to match caller
        'z_median_filled': z_median_filled,
        'y_mean_filled': y_mean_filled
    }


# ---------------------- Residual Boxplot (supports custom grouped data) ----------------------
def plot_residual_boxplot(fitting_results=None, output_path=None, custom_data=None, x_col=None, y_col=None,
                          show_plot=True, save_plot=True, save_path=None):
    """
    Plot residual boxplot: supports two modes
    1. Traditional mode based on fitting_results
    2. Grouped mode based on custom_data (group by x_col, compare y_col distribution across algorithms)
    """
    start_time = time.time()
    fig, ax = plt.subplots(figsize=(12, 7))

    # Mode 1: Use custom grouped data (priority)
    if custom_data is not None and x_col is not None and y_col is not None:
        # Parameter validation
        if pd is None:
            raise ImportError("pandas is required for grouped residual boxplots.")
        if not isinstance(custom_data, pd.DataFrame):
            raise ValueError("custom_data must be a pandas DataFrame")
        required_cols = ["Method", x_col, y_col]
        for col in required_cols:
            if col not in custom_data.columns:
                raise ValueError(f"custom_data is missing required column: {col}")

        # Filter invalid data
        valid_data = custom_data.dropna(subset=required_cols).copy()
        if valid_data.empty:
            print("Warning: no valid data for grouped residual boxplot")
            return None

        # Plot boxplot grouped by x_col and algorithm
        import seaborn as sns  # Lazy import to avoid unnecessary dependencies
        sns.boxplot(
            data=valid_data,
            x=x_col,
            y=y_col,
            hue="Method",
            ax=ax,
            palette=ALGORITHM_COLORS,  # Reuse algorithm colors
            showfliers=True,
            boxprops=dict(alpha=0.7)
        )
        ax.set_title(f"{y_col} by {x_col} Group Comparison", fontsize=14, pad=15)
        ax.legend(title="Algorithm", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Mode 2: Traditional mode (based on fitting_results)
    elif fitting_results is not None:
        valid_residuals = []
        method_names = []
        for fr in fitting_results:
            if "residuals" in fr and "inliers" in fr and fr["method"]:
                # Use absolute residuals of inliers only
                inlier_mask = coerce_inlier_mask(fr["inliers"], len(fr["residuals"]), method_name=fr["method"])
                inlier_residuals = np.abs(fr["residuals"][inlier_mask])
                if len(inlier_residuals) > 0:
                    valid_residuals.append(inlier_residuals)
                    method_names.append(fr["method"])

        if not valid_residuals:
            print("Warning: no valid residual data for boxplot")
            return None

        # Plot traditional boxplot
        bp = ax.boxplot(valid_residuals, patch_artist=True, labels=method_names)
        for patch, method in zip(bp['boxes'], method_names):
            patch.set_facecolor(ALGORITHM_COLORS.get(method, "#cccccc"))
            patch.set_alpha(0.7)
        ax.set_title("Inlier Residual Distribution Across Algorithms", fontsize=14, pad=15)

    else:
        raise ValueError("Must provide either fitting_results or custom_data+x_col+y_col")

    # General chart settings
    ax.set_xlabel(x_col if x_col else "Algorithm", fontsize=12)
    ax.set_ylabel(y_col if y_col else "Residual (m)", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()

    # 保存图像 (优先使用新参数save_path，兼容旧参数output_path)
    if save_plot:
        final_save_path = save_path if save_path is not None else output_path
        if not final_save_path:
            print("Warning: save_plot=True but save_path is None, skipping save")
        else:
            fig.savefig(final_save_path, dpi=1200, bbox_inches='tight')
            print(f"Residual boxplot saved to: {final_save_path}")

    # 显示图像
    if show_plot:
        plt.show()

    print(f"Residual boxplot plotting time: {time.time() - start_time:.2f}s")
    return fig


def get_color_map(method_names):
    """Generate algorithm-color mapping (prioritize predefined colors)"""
    color_map = {}
    for name in method_names:
        if name in ALGORITHM_COLORS:
            color_map[name] = ALGORITHM_COLORS[name]
        else:
            # Automatically generate colors as fallback
            sorted_names = sorted(method_names)
            cmap = plt.cm.tab10
            color_map[name] = cmap(sorted_names.index(name) % 10)
    return color_map


def plot_efficiency_comparison(data, algorithm_col=None, time_col='cpu_time',
                               xlabel='Algorithm', ylabel='CPU Time (s)',
                               title='Algorithm Efficiency Comparison',
                               truncation_ratio=5.0,
                               show_plot=True, save_plot=True, save_path=None):  # 极端值判断阈值（超过第二大值的倍数）
    """
    绘制带同步截断的效率对比柱状图（坐标轴与柱状图同时截断）

    参数:
        data: DataFrame，包含算法名称和时间数据
        algorithm_col: 算法名称列名
        time_col: 时间数据列名
        truncation_ratio: 极端值判断阈值（默认5倍于第二大值）
        show_plot: 是否显示图像（默认True）
        save_plot: 是否保存图像（默认True）
        save_path: 保存路径（仅save_plot=True时生效）
    """
    fig = plt.figure(figsize=(10, 6))

    # 1. 数据预处理：过滤无效值并计算平均值
    valid_data = data.dropna(subset=[algorithm_col, time_col])
    valid_data = valid_data[valid_data[time_col] != 'Failed']  # 排除失败案例
    valid_data[time_col] = valid_data[time_col].astype(float)  # 确保数值类型
    mean_times = valid_data.groupby(algorithm_col)[time_col].mean().sort_values()

    if mean_times.empty:
        print("Warning: no valid data for efficiency comparison")
        return fig

    algorithms = mean_times.index.tolist()
    times = mean_times.values

    # 2. 检测极端值（是否需要截断）
    if len(times) < 2:
        # 数据量不足，无需截断
        ax = fig.add_subplot(111)
        bars = ax.bar(algorithms, times, color=[ALGORITHM_COLORS.get(alg, '#aaaaaa') for alg in algorithms])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        plt.tight_layout()
    else:
        # 排序后取第二大值作为参考
        sorted_times = sorted(times)
        second_max = sorted_times[-2]
        max_val = sorted_times[-1]
        need_truncation = max_val > truncation_ratio * second_max  # 极端值判断

        if not need_truncation:
            # 无极端值，正常绘制
            ax = fig.add_subplot(111)
            bars = ax.bar(algorithms, times, color=[ALGORITHM_COLORS.get(alg, '#aaaaaa') for alg in algorithms])
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            ax.set_title(title)
            # 标注数值
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
            plt.tight_layout()
        else:
            # 3. 有极端值，执行同步截断
            # 截断位置：略高于第二大值（留出视觉空间）
            truncation_pos = second_max * 1.1
            # 上段Y轴范围（简化展示极端值）
            upper_ylim = (truncation_pos * 1.2, max_val * 1.1)

            # 创建上下两个子图（共享X轴）
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)  # 下段高度是上段3倍
            ax_lower = fig.add_subplot(gs[0])  # 下段：常规数据
            ax_upper = fig.add_subplot(gs[1], sharex=ax_lower)  # 上段：极端值截断部分
            plt.setp(ax_upper.get_xticklabels(), visible=False)  # 上段隐藏X轴标签

            # 4. 绘制下段柱状图（常规数据 + 极端值截断部分）
            lower_bars = []
            for alg, t in zip(algorithms, times):
                color = ALGORITHM_COLORS.get(alg, '#aaaaaa')
                if t <= truncation_pos:
                    # 常规值：完整绘制
                    bar = ax_lower.bar(alg, t, color=color)
                else:
                    # 极端值：下段绘制到截断位置
                    bar = ax_lower.bar(alg, truncation_pos, color=color)
                lower_bars.append(bar)

            # 5. 绘制上段柱状图（仅极端值的“截断后部分”）
            for alg, t in zip(algorithms, times):
                color = ALGORITHM_COLORS.get(alg, '#aaaaaa')
                if t > truncation_pos:
                    # 上段高度 = 实际值 - 截断位置（按比例压缩）
                    upper_height = (t - truncation_pos) * (upper_ylim[1] - upper_ylim[0]) / (max_val - truncation_pos)
                    ax_upper.bar(alg, upper_height, bottom=upper_ylim[0], color=color)

            # 6. 设置坐标轴范围与截断符号
            # 下段Y轴：0到截断位置
            ax_lower.set_ylim(0, truncation_pos)
            # 上段Y轴：截断位置到极端值（压缩显示）
            ax_upper.set_ylim(upper_ylim)

            # 在两段之间添加截断符号“//”
            ax_lower.text(0.02, 0.98, '//', transform=ax_lower.transAxes,
                          fontsize=12, va='top', ha='left', color='gray')

            # 7. 标注极端值的实际数值
            for i, (alg, t) in enumerate(zip(algorithms, times)):
                if t > truncation_pos:
                    # 在对应柱子上方标注真实值
                    ax_upper.text(i, upper_ylim[1] * 1.02, f'{t:.2f}',
                                  ha='center', va='bottom', fontsize=9, color='crimson')

            # 8. 美化图表
            ax_lower.set_ylabel(ylabel)
            ax_upper.set_title(title, pad=20)
            ax_lower.grid(axis='y', alpha=0.3)
            ax_upper.grid(axis='y', alpha=0.3)
            fig.text(0.5, 0.04, xlabel, ha='center', fontsize=12)  # 共享X轴标签

            plt.tight_layout()

    # 保存图像
    if save_plot:
        if not save_path:
            print("Warning: save_plot=True but save_path is None, skipping save")
        else:
            fig.savefig(save_path, dpi=1200, bbox_inches='tight')
            print(f"Efficiency comparison plot saved to: {save_path}")

    # 显示图像
    if show_plot:
        plt.show()

    return fig


# ---------------------- 13. Stability Comparison Plot (enhanced validity check, avoid NaN) ----------------------
def plot_stability_comparison(fitting_results, results_df, output_path=None,
                              show_plot=True, save_plot=True, save_path=None):
    """Optimized stability comparison plot: check baseline validity, filter invalid algorithms"""
    start_time = time.time()
    # Filter valid algorithms (non-empty normal vector/intercept, and result not "Failed")
    valid_methods = [
        fr for fr in fitting_results
        if fr.get("n_est") is not None and fr.get("d_est") is not None
           and not np.isnan(fr["n_est"]).any() and not np.isnan(fr["d_est"])
           and fr["method"] in results_df[results_df["Normal Vector (nx,ny,nz)"] != "Failed"]["Method"].values
    ]
    if len(valid_methods) < 2:
        print("Warning: need at least 2 valid methods for stability plot")
        return None

    method_names = [fr["method"] for fr in valid_methods]
    # Select first valid algorithm as baseline
    baseline_n = valid_methods[0]["n_est"]
    baseline_d = valid_methods[0]["d_est"]

    # Calculate cosine similarity of normal vectors (1 - cosine distance) and intercept difference
    cos_sim = []
    d_diff = []
    for fr in valid_methods:
        n = fr["n_est"]
        # Cosine similarity (avoid division by zero)
        sim = 1 - cosine(n, baseline_n) if np.linalg.norm(n) > 0 and np.linalg.norm(baseline_n) > 0 else 0
        cos_sim.append(sim)
        # Absolute intercept difference
        d_diff.append(np.abs(fr["d_est"] - baseline_d))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Normal vector similarity (left Y-axis)
    min_sim = min(cos_sim)
    y1_min = max(0.95, min_sim * 0.98)  # Limit range to highlight differences
    y1_max = 1.005
    ax1.plot(
        method_names, cos_sim,
        marker='o', color='#2ecc71', linestyle='-', linewidth=2.5,
        markersize=8, markerfacecolor='white', markeredgewidth=2,
        label='Normal Vector Similarity'
    )
    ax1.set_ylabel('Normal Vector Similarity', fontsize=12, color='#27ae60')
    ax1.set_ylim(y1_min, y1_max)
    ax1.tick_params(axis='y', labelcolor='#27ae60', labelsize=10)
    ax1.set_xticks(range(len(method_names)))
    ax1.set_xticklabels(method_names, rotation=45, ha='right', fontsize=10)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax1.set_axisbelow(True)

    # Intercept absolute difference (right Y-axis)
    ax2 = ax1.twinx()
    ax2.plot(
        method_names, d_diff,
        marker='s', color='#e74c3c', linestyle='--', linewidth=2.5,
        markersize=8, markerfacecolor='white', markeredgewidth=2,
        label='Intercept Absolute Difference'
    )
    ax2.set_ylabel('Intercept Absolute Difference (m)', fontsize=12, color='#c0392b')
    ax2.tick_params(axis='y', labelcolor='#c0392b', labelsize=10)
    # Adjust Y-axis range (avoid zero compression)
    max_diff = max(d_diff) if d_diff else 0
    ax2.set_ylim(-max_diff * 0.1, max_diff * 1.1)

    # Legend and title
    ax1.set_title('Stability Comparison: Vector Consistency & Position Deviation',
                  fontsize=14, pad=20)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='lower center', ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.25), frameon=True, edgecolor='lightgray')

    plt.tight_layout()

    # 保存图像 (优先使用新参数save_path，兼容旧参数output_path)
    if save_plot:
        final_save_path = save_path if save_path is not None else output_path
        if not final_save_path:
            print("Warning: save_plot=True but save_path is None, skipping save")
        else:
            fig.savefig(final_save_path, dpi=1200, bbox_inches='tight')
            print(f"Stability plot saved to: {final_save_path}")

    # 显示图像
    if show_plot:
        plt.show()

    print(f"Stability comparison plot plotting time: {time.time() - start_time:.2f}s")
    return fig


# ---------------------- 15. Roughness + Elevation Comprehensive Summary Plot (optimized layout, unified colors) ----------------------
def plot_roughness_elevation_summary(
        points, fitting_results, bin_width=1.0, grid_size=80,
        show_plot=True, save_plot=True, save_path=None
):
    """
    粗糙度+高程综合汇总图

    参数：
        points: 点云坐标 (N, 3)
        fitting_results: 算法结果列表，每个元素为{'name': str, 'residuals': (N,), 'n_est': (3,), 'd_est': scalar}
        bin_width: X轴分箱宽度
        grid_size: 热图插值网格大小
        show_plot: 是否显示图像（默认True）
        save_plot: 是否保存图像（默认True）
        save_path: 保存路径（仅save_plot=True时生效，如"summary.png"）
    返回：matplotlib的fig对象
    """
    start_time = time.time()
    # 过滤有效算法
    valid_results = [
        fr for fr in fitting_results
        if "method" in fr and "residuals" in fr and "n_est" in fr and "d_est" in fr
           and len(fr["residuals"]) == len(points)
           and not np.isnan(fr["n_est"]).any() and not np.isnan(fr["d_est"])
    ]
    if not valid_results:
        print("Warning: no valid results for summary plot")
        return None

    # 布局与颜色范围
    n_algorithms = len(valid_results)
    n_cols = min(4, n_algorithms)
    n_rows = (n_algorithms + n_cols - 1) // n_cols
    fig_width = max(9.5, 3.55 * n_cols + 0.9)
    fig_height = max(5.8, 4.15 * n_rows + 0.8)
    all_residuals = np.concatenate([np.abs(fr["residuals"]) for fr in valid_results])
    vmin, vmax = np.percentile(all_residuals, [1, 99])

    # 初始化画布
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(
        n_rows * 2,
        n_cols + 1,
        width_ratios=[1.0] * n_cols + [0.06],
        wspace=0.28,
        hspace=0.62,
    )
    legend_handles = None
    legend_labels = None

    # 绘制每个算法的子图
    for i, fr in enumerate(valid_results):
        method_name = fr["method"]
        residuals = fr["residuals"]
        n_est, d_est = fr["n_est"], fr["d_est"]
        row = i // n_cols
        col = i % n_cols

        # 1. 粗糙度热图子图
        ax_heat = fig.add_subplot(gs[row * 2, col])
        interp_ratio = min(50000 / len(points), 1.0)
        interp_points, interp_indices = downsample_points(points, ratio=interp_ratio)
        interp_residuals = residuals[interp_indices]
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        xi = np.linspace(x_min, x_max, grid_size)
        yi = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(xi, yi)
        residual_grid = griddata(
            (interp_points[:, 0], interp_points[:, 1]),
            np.abs(interp_residuals),
            (X, Y),
            method='linear',
            fill_value=0
        )
        im = ax_heat.imshow(
            residual_grid,
            extent=[x_min, x_max, y_min, y_max],
            cmap='RdYlGn_r',
            aspect='auto',
            origin='lower',
            vmin=vmin,
            vmax=vmax
        )
        ax_heat.set_title(f"{method_name} - Roughness", fontsize=10, pad=8)
        ax_heat.set_xlabel("X (m)", fontsize=8)
        ax_heat.tick_params(labelsize=7, length=2.5)
        ax_heat.spines["top"].set_visible(False)
        ax_heat.spines["right"].set_visible(False)
        ax_heat.set_yticks([]) if col != 0 else ax_heat.set_ylabel("Y (m)", fontsize=8)

        # 2. 高程剖面子图
        ax_elev = fig.add_subplot(gs[row * 2 + 1, col])
        cache = preprocess_elevation_bins(points, bin_width=bin_width)
        y_mean = np.mean(points[:, 1])
        z_plane_per_bin = -(n_est[0] * cache['x_bin_centers'] + n_est[1] * y_mean + d_est) / n_est[2]
        line_original = ax_elev.plot(
            cache['x_bin_centers'], cache['z_median_filled'],
            'b-', linewidth=1.2, alpha=0.5,
            label="Original Median Trend"
        )[0]
        line_fitted = ax_elev.plot(
            cache['x_bin_centers'], z_plane_per_bin,
            'r--', linewidth=1.5, alpha=0.5,
            label="Fitted Plane Trend"
        )[0]
        line_median = ax_elev.axhline(y=np.median(points[:, 2]), color='gray', linestyle=':', alpha=0.5,
                                      label='Original Median')
        line_plane_mean = ax_elev.axhline(y=np.mean(z_plane_per_bin), color='orange', linestyle=':', alpha=0.5,
                                          label='Plane Mean')
        ax_elev.set_title(f"{method_name} - Elevation", fontsize=10, pad=8)
        ax_elev.set_xlabel("X (m)", fontsize=8)
        ax_elev.tick_params(labelsize=7, length=2.5)
        ax_elev.spines["top"].set_visible(False)
        ax_elev.spines["right"].set_visible(False)
        ax_elev.grid(alpha=0.3, linestyle='--')
        ax_elev.set_yticks([]) if col != 0 else ax_elev.set_ylabel("Elevation (m)", fontsize=8)
        if n_algorithms == 1:
            ax_elev.legend(
                fontsize=7.2,
                loc='upper left',
                frameon=False,
                ncol=2,
                columnspacing=1.0,
                handlelength=2.2,
            )
        if legend_handles is None:
            legend_handles = [line_original, line_fitted, line_median, line_plane_mean]
            legend_labels = [handle.get_label() for handle in legend_handles]

    # 共享色条与整体标题
    cbar_ax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Absolute Residual (m)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle("Roughness and Elevation Summary", fontsize=14, y=0.988, fontweight="bold")
    if legend_handles and n_algorithms > 1:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.47, 0.945),
            ncol=min(4, len(legend_handles)),
            frameon=False,
            fontsize=8,
            handlelength=2.4,
            columnspacing=1.3,
        )
    fig.subplots_adjust(
        left=0.06,
        right=0.96,
        bottom=0.06,
        top=0.91 if n_algorithms == 1 else 0.86,
    )

    # 保存图像
    if save_plot:
        if not save_path:
            print("Warning: save_plot=True but save_path is None, skipping save")
        else:
            fig.savefig(save_path, dpi=1200, bbox_inches='tight')
            print(f"Summary plot saved to: {save_path}")

    # 显示图像
    if show_plot:
        plt.show()

    print(f"Summary plot time: {time.time() - start_time:.2f}s")
    return fig

