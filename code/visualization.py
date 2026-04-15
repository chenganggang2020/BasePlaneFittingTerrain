
import matplotlib
matplotlib.use("Agg", force=True)
from matplotlib import cm
import os
import numpy as np
try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional dependency
    sns = None
from constants import ALGORITHM_COLORS, ALGORITHM_MARKERS
from utils import downsample_data

# 导入外部可视化函数
from visualizer import (
    plot_common_3d_terrain,
    plot_common_outlier_dist,
    plot_roughness_heatmap,
    plot_elevation_profile
)


def plot_plane_fitting_result(points, n, d, inliers, method_name):
    """可视化单个算法的平面拟合结果，返回figure对象"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[inliers, 0], points[inliers, 1], points[inliers, 2],
               c='green', s=1, label='Inliers')
    ax.scatter(points[~inliers, 0], points[~inliers, 1], points[~inliers, 2],
               c='red', s=1, label='Outliers')

    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    x_range = np.linspace(x_min, x_max, 50)
    y_range = np.linspace(y_min, y_max, 50)
    X, Y = np.meshgrid(x_range, y_range)

    if np.abs(n[2]) < 1e-8:
        Z = np.zeros_like(X)
    else:
        Z = (-n[0] * X - n[1] * Y - d) / n[2]

    ax.plot_surface(X, Y, Z, alpha=0.5, color='blue', cmap=cm.coolwarm)

    ax.set_title(f'Plane Fitting - {method_name}', fontsize=12)
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    ax.legend()

    x_range_val = x_max - x_min
    y_range_val = y_max - y_min
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    z_range_val = z_max - z_min

    max_range = max(x_range_val, y_range_val, z_range_val)
    ax.set_box_aspect([x_range_val / max_range, y_range_val / max_range, z_range_val / max_range])

    ax.set_xlim(x_min + x_range_val / 2 - max_range / 2, x_min + x_range_val / 2 + max_range / 2)
    ax.set_ylim(y_min + y_range_val / 2 - max_range / 2, y_min + y_range_val / 2 + max_range / 2)
    ax.set_zlim(z_min + z_range_val / 2 - max_range / 2, z_min + z_range_val / 2 + max_range / 2)

    return fig


def plot_performance_comparison(results):
    """可视化各算法的性能对比，返回figure对象"""
    methods = [item['Method'] for item in results]
    cpu_times = [float(item['CPU Time (s)']) if item['CPU Time (s)'] != 'Failed' else 0 for item in results]
    inlier_ratios = [float(item['Inlier Ratio'].strip('%')) / 100 if item['Inlier Ratio'] != 'Failed' else 0 for item in
                     results]
    mean_residuals = [float(item['Mean Inlier Residual (m)']) if item['Mean Inlier Residual (m)'] != 'N/A' and item[
        'Mean Inlier Residual (m)'] != 'Failed' else 0 for item in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.bar(methods, cpu_times, color='skyblue')
    ax1.set_title('CPU Time Comparison (seconds)', fontsize=12)
    ax1.set_xlabel('Algorithm', fontsize=10)
    ax1.set_ylabel('CPU Time (s)', fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    ax2.plot(methods, inlier_ratios, 'o-', color='green', label='Inlier Ratio')
    ax2.set_xlabel('Algorithm', fontsize=10)
    ax2.set_ylabel('Inlier Ratio', fontsize=10, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    ax3 = ax2.twinx()
    ax3.plot(methods, mean_residuals, 's-', color='red', label='Mean Residual')
    ax3.set_ylabel('Mean Inlier Residual (m)', fontsize=10, color='red')
    ax3.tick_params(axis='y', labelcolor='red')

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')

    ax2.set_title('Inlier Ratio vs Mean Residual', fontsize=12)

    plt.tight_layout()
    return fig


def generate_monte_carlo_plots(agg_df, all_methods, sto_names, comparison_dir):
    """生成蒙特卡洛实验统计图表（处理RWTLS内存不足情况）"""
    print("\n=== Generating Monte Carlo statistical plots ===")
    numeric_cols = [
        "Number of Inliers", "Mean Inlier Residual (m)", "CPU Time (s)",
        "Accuracy", "False Negative Rate", "False Positive Rate"
    ]
    # 确保数值列类型正确
    for col in numeric_cols:
        agg_df[col] = pd.to_numeric(agg_df[col], errors='coerce')

    # 1. 趋势图
    for metric in numeric_cols:
        try:
            plt.figure(figsize=(12, 6))
            for method in all_methods:
                method_data = agg_df[agg_df["Method"] == method]
                if method == "RWTLS" and method_data[metric].isna().all():
                    # RWTLS内存不足：灰色虚线占位
                    plt.plot(
                        method_data["Trial Number"], method_data[metric],
                        label=method + " (Out of memory)",
                        color="#aaaaaa", linestyle="--", linewidth=2,
                        marker="", markersize=0
                    )
                else:
                    # 正常绘图
                    plt.plot(
                        method_data["Trial Number"], method_data[metric],
                        label=method,
                        color=ALGORITHM_COLORS[method], linewidth=1,
                        marker=ALGORITHM_MARKERS[method], markersize=2
                    )
            plt.title(f"Monte Carlo: {metric} Trend Across Trials", fontsize=12, fontweight='bold')
            plt.xlabel("Trial Number", fontsize=10)
            plt.ylabel(metric, fontsize=10)
            plt.grid(linestyle='--', alpha=0.7)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            plt.tight_layout()
            trend_path = os.path.join(comparison_dir, f"montecarlo_trend_{metric.replace(' ', '_')}.png")
            plt.savefig(trend_path, dpi=1200)
            plt.show()
            print(f"Monte Carlo trend plot saved: {trend_path}")
        except Exception as e:
            print(f"❌ Failed to generate trend plot ({metric}): {e}")

    # 2. 均值±标准差图
    stats = agg_df.groupby("Method", observed=True)[numeric_cols].agg(['mean', 'std']).reindex(all_methods)
    for metric in numeric_cols:
        try:
            plt.figure(figsize=(10, 6))
            x = np.arange(len(all_methods))
            for i, method in enumerate(all_methods):
                if method == "RWTLS" and pd.isna(stats.loc[method, (metric, 'mean')]):
                    # RWTLS内存不足：灰色空柱+标注
                    plt.bar(x[i], 0, width=0.6, color="#aaaaaa", edgecolor='black', linewidth=1,
                            label=method + " (Out of memory)" if i == 0 else "")
                    plt.text(x[i], 0.05, "OOM", ha='center', va='bottom', fontsize=8)
                else:
                    mean = stats.loc[method, (metric, 'mean')]
                    std = stats.loc[method, (metric, 'std')]
                    plt.bar(x[i], mean, yerr=std, capsize=6, width=0.6,
                            color=ALGORITHM_COLORS[method], edgecolor='black', linewidth=1,
                            label=method if i == 0 else "")
            plt.xticks(x, all_methods, rotation=45, ha='right', fontsize=9)
            plt.title(f"Monte Carlo: {metric} (Mean ± Std)", fontsize=12, fontweight='bold')
            plt.ylabel(metric, fontsize=10)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            plt.tight_layout()
            mean_std_path = os.path.join(comparison_dir, f"montecarlo_mean_std_{metric.replace(' ', '_')}.png")
            plt.savefig(mean_std_path, dpi=1200)
            plt.show()
            print(f"Monte Carlo mean±std plot saved: {mean_std_path}")
        except Exception as e:
            print(f"❌ Failed to generate mean±std plot ({metric}): {e}")

    # 3. 随机算法的箱线图（CPU时间分布）
    try:
        plt.figure(figsize=(10, 6))
        valid_sto = agg_df[agg_df["Method"].isin(sto_names)].copy()
        valid_sto["Method"] = pd.Categorical(valid_sto["Method"], categories=sto_names, ordered=True)
        sns.boxplot(
            data=valid_sto, x="Method", y="CPU Time (s)",
            palette=ALGORITHM_COLORS, showfliers=True
        )
        plt.title("Monte Carlo: CPU Time Distribution (Stochastic Algorithms)", fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.ylabel("CPU Time (s)", fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        boxplot_path = os.path.join(comparison_dir, "montecarlo_stochastic_cpu_boxplot.png")
        plt.savefig(boxplot_path, dpi=1200)
        plt.show()
        print(f"Monte Carlo stochastic boxplot saved: {boxplot_path}")
    except Exception as e:
        print(f"❌ Failed to generate stochastic boxplot: {e}")


def generate_single_trial_plots(points, final_fitting_results, all_methods, root_dir, method_plots_dir):
    """生成单轮试验的可视化图表"""
    print("\n=== Generating single-trial result plots ===")
    # 过滤有效拟合结果
    valid_fitting_results = [
        fr for fr in final_fitting_results
        if fr["n_est"] is not None and fr["d_est"] is not None
    ]

    # 3D地形图
    if valid_fitting_results:
        common_3d_fig = plot_common_3d_terrain(points, valid_fitting_results)
        common_3d_path = os.path.join(root_dir, "common_3d_terrain.png")
        common_3d_fig.savefig(common_3d_path, dpi=1200, bbox_inches='tight')
        plt.show()
        print(f"Common 3D terrain plot saved: {common_3d_path}")
    else:
        print("Warning: no valid fitting results for 3D terrain plot, skipping.")

    # 异常值分布图（基于LS算法）
    ls_result = next((fr for fr in final_fitting_results if fr["method"] == "LS"), None)
    if ls_result:
        common_outlier_fig = plot_common_outlier_dist(points, ls_result["inliers"], "X (m)", "Y (m)")
        common_outlier_path = os.path.join(root_dir, "common_outlier_distribution.png")
        common_outlier_fig.savefig(common_outlier_path, dpi=1200, bbox_inches='tight')
        plt.show()
        print(f"Common outlier distribution plot saved: {common_outlier_path}")

    # 算法专属图表
    print("\n=== Generating method-specific plots ===")
    downsample_step = 20
    for fr in final_fitting_results:
        method_name = fr["method"]
        # 跳过无数据的RWTLS
        if method_name == "RWTLS" and fr["n_est"] is None:
            print(f"Warning: skipping {method_name} specific plots because it ran out of memory.")
            continue

        n_est, d_est = fr["n_est"], fr["d_est"]
        residuals = fr["residuals"]

        # 降采样处理
        if points.shape[0] > 10000:
            downsampled_points, downsampled_residuals = downsample_data(
                points, residuals, step=downsample_step
            )
            print(f"📉 Downsampled data for {method_name}: {len(downsampled_points)}/{len(points)} points")
        else:
            downsampled_points, downsampled_residuals = points, residuals

        # 粗糙度热图
        rough_path = os.path.join(method_plots_dir, f"{method_name}_roughness.png")
        plot_roughness_heatmap(
            downsampled_points, np.abs(downsampled_residuals),
            method_name, show_plot=False, save_path=rough_path
        )

        # 高程剖面图
        elev_path = os.path.join(method_plots_dir, f"{method_name}_elevation.png")
        plot_elevation_profile(
            downsampled_points, n_est, d_est, method_name,
            show_plot=False, bin_width=1.0, save_path=elev_path
        )
