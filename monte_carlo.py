import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    class _TqdmFallback:
        def __init__(self, iterable=None, total=None, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter(())

        def update(self, n=1):
            return None

        def close(self):
            return None

    def tqdm(iterable=None, *args, **kwargs):
        return _TqdmFallback(iterable=iterable, **kwargs)
from export_utils import export_to_cloudcompare, export_to_matlab
from utils import load_point_cloud, downsample_data, extract_outlier_ratio, parse_array_from_string
from metrics import calculate_detection_metrics, calculate_angle_between_vectors,calculate_d_est_difference
from algorithm_runner import get_algorithm_categories
from visualization import generate_monte_carlo_plots, generate_single_trial_plots
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from constants import ALGORITHM_COLORS, ALGORITHM_MARKERS, PLOT_CONFIG
from visualizer import plot_roughness_heatmap, plot_common_3d_terrain, plot_elevation_profile, \
    plot_elevation_trend_comparison, plot_roughness_elevation_summary
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, LogFormatterExponent
from safety_evaluation import (
    compare_safety_maps,
    evaluate_plane_safety,
    plot_safety_coefficient_map,
    plot_window_score_map,
)
from risk_probability_map import (
    compute_risk_probability_map,
    plot_feature_maps,
    plot_risk_fusion_maps,
)
from joint_risk_safety import (
    compare_joint_safe_maps,
    evaluate_joint_risk_safety,
    plot_joint_fusion_maps,
    plot_joint_safe_window_map,
)
from tabular_results import (
    filter_numeric_rows,
    mean_ignore_nan,
    merge_rows_on_key,
    read_rows_from_csv,
    to_float,
    write_matrix_to_csv,
    write_rows_to_csv,
    write_single_row_csv,
)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = True


def _extract_metric_pairs(rows, metric, method=None, x_key="Outlier Ratio (%)"):
    pairs = []
    for row in rows:
        if method is not None and row.get("Method") != method:
            continue
        x_value = to_float(row.get(x_key))
        y_value = to_float(row.get(metric))
        if np.isnan(x_value) or np.isnan(y_value):
            continue
        pairs.append((x_value, y_value))
    pairs.sort(key=lambda item: item[0])
    return pairs


def _sanitize_metric_name(metric):
    return metric.replace(" ", "_").replace("/", "_")


def _identity_eps_mapping(outlier_ratio):
    return outlier_ratio


def _resolve_max_workers(max_workers):
    if max_workers is None:
        return 1
    if isinstance(max_workers, str):
        if max_workers.lower() == "auto":
            max_workers = 0
        else:
            max_workers = int(max_workers)
    if int(max_workers) <= 0:
        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count - 1)
    return max(1, int(max_workers))


def _build_parallel_executor(max_workers):
    try:
        return ProcessPoolExecutor(max_workers=max_workers), "process"
    except (PermissionError, OSError) as exc:
        print(f"Warning: process pool unavailable ({exc}); falling back to thread pool.")
        return ThreadPoolExecutor(max_workers=max_workers), "thread"


BOUNDED_SCORE_METRICS = {
    "Accuracy",
    "False Negative Rate",
    "False Positive Rate",
    "Mean Safety Coefficient",
    "Best Window Mean Safety",
    "Mean Joint Hazard",
    "High Joint Hazard Ratio",
    "Mean Joint Safe Score",
    "Best Joint Window Score",
    "Joint Map Accuracy",
    "Joint Map Precision",
    "Joint Map Recall",
    "Joint Map F1",
    "Risk Map Accuracy",
    "Risk Map Precision",
    "Risk Map Recall",
    "Risk Map F1",
    "Mean Risk Probability",
    "Max Risk Probability",
    "High Risk Ratio",
}

LOG_CANDIDATE_METRICS = {
    "CPU Time (s)",
    "Mean Inlier Residual (m)",
    "Interception Difference (m)",
    "Normal Deviation (deg)",
    "Safety Map Abs Difference Sum",
    "Joint Safe Map Abs Difference Sum",
}


def _metric_is_bounded(metric):
    return metric in BOUNDED_SCORE_METRICS


def _should_use_log_scale(metric, values):
    if metric not in LOG_CANDIDATE_METRICS:
        return False
    if any(value <= 0 for value in values):
        return False
    positive = [value for value in values if value > 0]
    if len(positive) < 3:
        return False
    dynamic_range = max(positive) / max(min(positive), 1e-12)
    return dynamic_range >= 20


def _compute_linear_limits(values, bounded=False, lower_bound=None, upper_bound=None):
    if not values:
        return None

    y_min = min(values)
    y_max = max(values)
    if np.isclose(y_min, y_max):
        pad = max(abs(y_min) * 0.05, 1e-3)
        y_min -= pad
        y_max += pad
    else:
        pad = (y_max - y_min) * 0.08
        y_min -= pad
        y_max += pad

    if bounded:
        y_min = 0.0 if lower_bound is None else max(lower_bound, y_min)
        y_max = 1.02 if upper_bound is None else min(upper_bound, y_max)
    else:
        if lower_bound is not None:
            y_min = max(lower_bound, y_min)
        if upper_bound is not None:
            y_max = min(upper_bound, y_max)
    return y_min, y_max


def _style_journal_axis(ax, metric, use_log_scale=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.15)
    ax.tick_params(axis="both", which="major", labelsize=9, width=0.8, length=4)
    ax.tick_params(axis="both", which="minor", width=0.6, length=2)

    if use_log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(LogFormatterExponent())
    elif _metric_is_bounded(metric):
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    else:
        ax.yaxis.set_major_formatter(ScalarFormatter())


def _analyze_point_cloud_worker(analyzer_kwargs, task_kwargs):
    analyzer = PointCloudAnalyzer(**analyzer_kwargs)
    return analyzer.analyze_point_cloud(**task_kwargs)


class PointCloudAnalyzer:
    """
    鐐逛簯鍒嗘瀽鏍稿績绫伙細灏佽鎵€鏈夌偣浜戝鐞嗘祦绋嬶紝鏀寔鍗曠偣浜戝垎鏋愬拰澶栫偣姣斾緥鎵弿
    """

    def __init__(
        self,
        tau=1.2,
        eps=0.8,
        initial_plane_slope=None,
        initial_plane_interception=None,
        export_formats=None,
        safety_height_threshold=0.3,
        safety_slope_threshold=10.0,
        safety_window_size=5,
        risk_window_size=5,
        risk_prior=0.35,
        risk_mrf_lambda=0.65,
        risk_mrf_iters=6,
        joint_risk_weight=0.55,
        joint_safety_weight=0.45,
        include_methods=None,
        exclude_methods=None,
        enable_risk_analysis=True,
        enable_safety_analysis=True,
        enable_joint_analysis=True,
    ):
        """
        鍒濆鍖栧垎鏋愬櫒
        :param tau: RANSAC闃堝€煎弬鏁?
        :param eps: 绠楁硶鍙傛暟
        :param initial_plane_normal: 鍒濆骞抽潰娉曞悜閲?
        :param export_formats: 瀵煎嚭鏍煎紡鍒楄〃锛堝["cloudcompare", "matlab"]锛?
        """
        self.tau = tau
        self.eps = eps
        self.has_reference_plane = initial_plane_slope is not None and initial_plane_interception is not None
        self.initial_plane_slope = initial_plane_slope if initial_plane_slope is not None else 0.0
        self.initial_plane_interception = initial_plane_interception if initial_plane_interception is not None else 0.0
        self.initial_plane_normal = np.array([-np.tan(np.radians(self.initial_plane_slope)), 0, 1], dtype=float)
        self.initial_plane_normal /= np.linalg.norm(self.initial_plane_normal)
        self.export_formats = export_formats
        self.safety_height_threshold = safety_height_threshold
        self.safety_slope_threshold = safety_slope_threshold
        self.safety_window_size = safety_window_size
        self.risk_window_size = risk_window_size
        self.risk_prior = risk_prior
        self.risk_mrf_lambda = risk_mrf_lambda
        self.risk_mrf_iters = risk_mrf_iters
        self.joint_risk_weight = joint_risk_weight
        self.joint_safety_weight = joint_safety_weight
        self.include_methods = list(include_methods) if include_methods is not None else None
        self.exclude_methods = list(exclude_methods) if exclude_methods is not None else []
        self.enable_risk_analysis = enable_risk_analysis
        self.enable_safety_analysis = enable_safety_analysis
        self.enable_joint_analysis = enable_joint_analysis

    def _build_analyzer_kwargs(self):
        return {
            "tau": self.tau,
            "eps": self.eps,
            "initial_plane_slope": self.initial_plane_slope if self.has_reference_plane else None,
            "initial_plane_interception": self.initial_plane_interception if self.has_reference_plane else None,
            "export_formats": self.export_formats,
            "safety_height_threshold": self.safety_height_threshold,
            "safety_slope_threshold": self.safety_slope_threshold,
            "safety_window_size": self.safety_window_size,
            "risk_window_size": self.risk_window_size,
            "risk_prior": self.risk_prior,
            "risk_mrf_lambda": self.risk_mrf_lambda,
            "risk_mrf_iters": self.risk_mrf_iters,
            "joint_risk_weight": self.joint_risk_weight,
            "joint_safety_weight": self.joint_safety_weight,
            "include_methods": self.include_methods,
            "exclude_methods": self.exclude_methods,
            "enable_risk_analysis": self.enable_risk_analysis,
            "enable_safety_analysis": self.enable_safety_analysis,
            "enable_joint_analysis": self.enable_joint_analysis,
        }

    def _get_algorithm_categories(self, eps):
        algo_categories = get_algorithm_categories(
            tau=self.tau,
            eps=eps,
            include_methods=self.include_methods,
            exclude_methods=self.exclude_methods,
        )
        if not algo_categories["all_methods"]:
            raise ValueError("No algorithms selected after applying include/exclude filters.")
        return algo_categories

    def _merge_method_summary(self, summary_rows, pc_results_csv):
        """Merge a per-method summary table back into the main per-point-cloud CSV."""
        if not os.path.exists(pc_results_csv) or not summary_rows:
            return
        base_rows = read_rows_from_csv(pc_results_csv)
        if not base_rows:
            return
        merged_rows = merge_rows_on_key(base_rows, summary_rows, key="Method")
        write_rows_to_csv(pc_results_csv, merged_rows)

    def _get_reference_plane(self):
        """Build the reference plane from the simulator slope/intercept when available."""
        if not self.has_reference_plane:
            return None
        n_ref = self.initial_plane_normal.copy()
        d_ref = -n_ref[2] * self.initial_plane_interception
        return n_ref, d_ref

    def _run_risk_probability_analysis(self, points, labels, pc_root_dir, point_cloud_name):
        """Run the large-area terrain risk probability pipeline for one point cloud."""
        risk_dir = os.path.join(pc_root_dir, "risk_probability")
        os.makedirs(risk_dir, exist_ok=True)

        try:
            risk_result = compute_risk_probability_map(
                points,
                labels=labels,
                window_size=self.risk_window_size,
                prior=self.risk_prior,
                mrf_lambda=self.risk_mrf_lambda,
                mrf_iters=self.risk_mrf_iters,
            )
            plot_feature_maps(
                risk_result,
                save_path=os.path.join(risk_dir, "terrain_features.png"),
                show_plot=False,
                dpi=300,
            )
            plot_risk_fusion_maps(
                risk_result,
                save_path=os.path.join(risk_dir, "risk_fusion.png"),
                show_plot=False,
                dpi=300,
            )

            summary_row = {
                "Point Cloud": point_cloud_name,
                "Mean Risk Probability": risk_result["mean_risk_probability"],
                "Max Risk Probability": risk_result["max_risk_probability"],
                "High Risk Ratio": risk_result["high_risk_ratio"],
                "Risk Prior": self.risk_prior,
                "MRF Lambda": self.risk_mrf_lambda,
                "MRF Iterations": self.risk_mrf_iters,
            }
            summary_row.update(risk_result["metrics"])
            write_single_row_csv(
                os.path.join(risk_dir, f"{point_cloud_name}_risk_summary.csv"),
                summary_row,
            )
            weight_rows = []
            for feature_name, weight in zip(
                risk_result["feature_names"],
                risk_result["covariance_weights"],
            ):
                weight_rows.append({
                    "Feature": feature_name,
                    "Covariance Corrected Weight": float(weight),
                })
            write_rows_to_csv(
                os.path.join(risk_dir, f"{point_cloud_name}_risk_feature_weights.csv"),
                weight_rows,
            )
            write_matrix_to_csv(
                os.path.join(risk_dir, f"{point_cloud_name}_feature_correlation.csv"),
                risk_result["correlation_matrix"],
                row_names=risk_result["feature_names"],
                col_names=risk_result["feature_names"],
                row_label="Feature",
            )
            print(f"Saved risk probability analysis to: {risk_dir}")
            return risk_result
        except Exception as exc:
            print(f"Warning: risk probability analysis failed for {point_cloud_name}: {exc}")
            return None

    def _collect_global_risk_summaries(self, pc_files_with_ratio, output_root):
        """Collect per-point-cloud risk summary CSVs into one global table."""
        rows = []
        for file_name, outlier_ratio in pc_files_with_ratio:
            point_cloud_name = os.path.splitext(file_name)[0]
            risk_csv = os.path.join(
                output_root,
                point_cloud_name,
                "risk_probability",
                f"{point_cloud_name}_risk_summary.csv",
            )
            if not os.path.exists(risk_csv):
                continue

            risk_summary_rows = read_rows_from_csv(risk_csv)
            if not risk_summary_rows:
                continue
            for risk_row in risk_summary_rows:
                combined_row = dict(risk_row)
                combined_row["File Name"] = file_name
                combined_row["Outlier Ratio"] = outlier_ratio
                combined_row["Outlier Ratio (%)"] = outlier_ratio * 100
                rows.append(combined_row)

        return rows

    def generate_global_risk_probability_plots(self, risk_rows, comparison_dir, dpi=1200, save_vector=False):
        """Generate point-cloud-level risk probability trend plots across outlier ratios."""
        if not risk_rows:
            return

        print("\n=== Generating global risk probability plots ===")
        numeric_cols = [
            "Mean Risk Probability",
            "Max Risk Probability",
            "High Risk Ratio",
            "Risk Map Accuracy",
            "Risk Map Precision",
            "Risk Map Recall",
            "Risk Map F1",
        ]
        available_cols = set()
        for row in risk_rows:
            available_cols.update(row.keys())
        numeric_cols = [col for col in numeric_cols if col in available_cols]

        os.makedirs(comparison_dir, exist_ok=True)
        for metric in numeric_cols:
            try:
                metric_pairs = _extract_metric_pairs(risk_rows, metric)
                if not metric_pairs:
                    continue
                x_values = [pair[0] for pair in metric_pairs]
                y_values = [pair[1] for pair in metric_pairs]

                fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"])
                ax.plot(
                    x_values,
                    y_values,
                    color="#bc4749",
                    linewidth=1.8,
                    marker="o",
                    markersize=4,
                )
                ax.set_title(f"Outlier Ratio vs {metric}", fontsize=PLOT_CONFIG["title_size"])
                ax.set_xlabel("Outlier Ratio (%)", fontsize=PLOT_CONFIG["font_size"])
                ax.set_ylabel(metric, fontsize=PLOT_CONFIG["font_size"])
                ax.grid(linestyle=PLOT_CONFIG["grid_style"], alpha=PLOT_CONFIG["grid_alpha"])

                avg_value = mean_ignore_nan(y_values)
                if not np.isnan(avg_value):
                    ax.text(
                        0.98,
                        0.98,
                        f"Avg: {avg_value:.3f}",
                        transform=ax.transAxes,
                        fontsize=PLOT_CONFIG["font_size"] * 0.8,
                        verticalalignment="top",
                        horizontalalignment="right",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="gray"),
                    )

                plt.tight_layout()
                plot_png = os.path.join(comparison_dir, f"risk_outlier_vs_{_sanitize_metric_name(metric)}.png")
                fig.savefig(plot_png, dpi=dpi, bbox_inches="tight")
                if save_vector:
                    plot_pdf = os.path.join(comparison_dir, f"risk_outlier_vs_{_sanitize_metric_name(metric)}.pdf")
                    fig.savefig(plot_pdf, format="pdf", bbox_inches="tight")
                plt.close(fig)
                print(f"Saved: {plot_png}")
            except Exception as exc:
                print(f"Failed to generate risk plot ({metric}): {exc}")

    def _run_safety_analysis(self, points, fitting_results, pc_root_dir, point_cloud_name, pc_results_csv):
        """Run paper-style safety coefficient analysis for every valid fitting result."""
        safety_dir = os.path.join(pc_root_dir, "safety_analysis")
        os.makedirs(safety_dir, exist_ok=True)

        reference_result = None
        reference_plane = self._get_reference_plane()
        if reference_plane is not None:
            try:
                reference_result = evaluate_plane_safety(
                    points,
                    reference_plane[0],
                    reference_plane[1],
                    h_threshold=self.safety_height_threshold,
                    theta_threshold=self.safety_slope_threshold,
                    window_size=self.safety_window_size,
                )
                plot_safety_coefficient_map(
                    reference_result,
                    title=f"{point_cloud_name} - Reference Safety Coefficient",
                    save_path=os.path.join(safety_dir, "reference_safety_coefficient.png"),
                    show_plot=False,
                    dpi=300,
                )
                plot_window_score_map(
                    reference_result,
                    title=f"{point_cloud_name} - Reference Safe Window Scores",
                    save_path=os.path.join(safety_dir, "reference_window_scores.png"),
                    show_plot=False,
                    dpi=300,
                )
            except Exception as exc:
                print(f"Warning: failed to compute reference safety map: {exc}")

        safety_rows = []
        safety_results_by_method = {}
        for fr in fitting_results:
            method_name = fr["method"]
            try:
                safety_result = evaluate_plane_safety(
                    points,
                    fr["n_est"],
                    fr["d_est"],
                    h_threshold=self.safety_height_threshold,
                    theta_threshold=self.safety_slope_threshold,
                    window_size=self.safety_window_size,
                )

                plot_safety_coefficient_map(
                    safety_result,
                    title=f"{point_cloud_name} - {method_name} Safety Coefficient",
                    save_path=os.path.join(safety_dir, f"{method_name}_safety_coefficient.png"),
                    show_plot=False,
                    dpi=300,
                )
                plot_window_score_map(
                    safety_result,
                    title=f"{point_cloud_name} - {method_name} Safe Window Scores",
                    save_path=os.path.join(safety_dir, f"{method_name}_window_scores.png"),
                    show_plot=False,
                    dpi=300,
                )

                best_window = safety_result["best_window"]
                row = {
                    "Method": method_name,
                    "Mean Safety Coefficient": safety_result["mean_safety"],
                    "Max Safety Coefficient": safety_result["max_safety"],
                    "Min Safety Coefficient": safety_result["min_safety"],
                    "Best Window Mean Safety": best_window["mean_score"] if best_window else np.nan,
                    "Best Window Min Safety": best_window["min_score"] if best_window else np.nan,
                    "Best Window Center X (m)": best_window["center_x"] if best_window else np.nan,
                    "Best Window Center Y (m)": best_window["center_y"] if best_window else np.nan,
                }
                if reference_result is not None:
                    row["Safety Map Abs Difference Sum"] = compare_safety_maps(
                        reference_result["safety_coefficient"],
                        safety_result["safety_coefficient"],
                    )
                safety_rows.append(row)
                safety_results_by_method[method_name] = safety_result
            except Exception as exc:
                print(f"Warning: safety analysis failed for {method_name}: {exc}")

        if not safety_rows:
            return {"reference_result": reference_result, "method_results": safety_results_by_method}

        safety_csv = os.path.join(pc_root_dir, f"{point_cloud_name}_safety_summary.csv")
        write_rows_to_csv(safety_csv, safety_rows)
        print(f"Saved safety summary to: {safety_csv}")
        self._merge_method_summary(safety_rows, pc_results_csv)

        return {"reference_result": reference_result, "method_results": safety_results_by_method}

    def _run_joint_analysis(self, risk_result, safety_analysis, pc_root_dir, point_cloud_name, pc_results_csv):
        """Fuse the macro risk map and the method-specific safety maps into joint decision outputs."""
        if not self.enable_joint_analysis:
            return
        if risk_result is None or safety_analysis is None:
            return

        safety_results = safety_analysis.get("method_results", {})
        if not safety_results:
            return

        joint_dir = os.path.join(pc_root_dir, "joint_risk_safety")
        os.makedirs(joint_dir, exist_ok=True)

        reference_joint_result = None
        reference_safety_result = safety_analysis.get("reference_result")
        if reference_safety_result is not None:
            try:
                reference_joint_result = evaluate_joint_risk_safety(
                    risk_result,
                    reference_safety_result,
                    risk_weight=self.joint_risk_weight,
                    safety_weight=self.joint_safety_weight,
                )
                plot_joint_fusion_maps(
                    reference_joint_result,
                    title=f"{point_cloud_name} - Reference Joint Decision",
                    save_path=os.path.join(joint_dir, "reference_joint_fusion.png"),
                    show_plot=False,
                    dpi=300,
                )
                plot_joint_safe_window_map(
                    reference_joint_result,
                    title=f"{point_cloud_name} - Reference Joint Safe Window",
                    save_path=os.path.join(joint_dir, "reference_joint_window.png"),
                    show_plot=False,
                    dpi=300,
                )
            except Exception as exc:
                print(f"Warning: reference joint analysis failed for {point_cloud_name}: {exc}")

        joint_rows = []
        for method_name, safety_result in safety_results.items():
            try:
                joint_result = evaluate_joint_risk_safety(
                    risk_result,
                    safety_result,
                    risk_weight=self.joint_risk_weight,
                    safety_weight=self.joint_safety_weight,
                )
                plot_joint_fusion_maps(
                    joint_result,
                    title=f"{point_cloud_name} - {method_name} Joint Decision",
                    save_path=os.path.join(joint_dir, f"{method_name}_joint_fusion.png"),
                    show_plot=False,
                    dpi=300,
                )
                plot_joint_safe_window_map(
                    joint_result,
                    title=f"{point_cloud_name} - {method_name} Joint Safe Window",
                    save_path=os.path.join(joint_dir, f"{method_name}_joint_window.png"),
                    show_plot=False,
                    dpi=300,
                )

                best_window = joint_result["best_window"]
                row = {
                    "Method": method_name,
                    "Mean Joint Hazard": joint_result["mean_joint_hazard"],
                    "Max Joint Hazard": joint_result["max_joint_hazard"],
                    "High Joint Hazard Ratio": joint_result["high_joint_hazard_ratio"],
                    "Mean Joint Safe Score": joint_result["mean_joint_safe_score"],
                    "Max Joint Safe Score": joint_result["max_joint_safe_score"],
                    "Best Joint Window Score": best_window["window_score"] if best_window else np.nan,
                    "Best Joint Window Mean Score": best_window["mean_score"] if best_window else np.nan,
                    "Best Joint Window Min Score": best_window["min_score"] if best_window else np.nan,
                    "Best Joint Window Center X (m)": best_window["center_x"] if best_window else np.nan,
                    "Best Joint Window Center Y (m)": best_window["center_y"] if best_window else np.nan,
                }
                row.update(joint_result["metrics"])
                if reference_joint_result is not None:
                    row["Joint Safe Map Abs Difference Sum"] = compare_joint_safe_maps(
                        reference_joint_result,
                        joint_result,
                    )
                joint_rows.append(row)
            except Exception as exc:
                print(f"Warning: joint analysis failed for {method_name}: {exc}")

        if not joint_rows:
            return

        joint_csv = os.path.join(pc_root_dir, f"{point_cloud_name}_joint_summary.csv")
        write_rows_to_csv(joint_csv, joint_rows)
        print(f"Saved joint risk-safety summary to: {joint_csv}")
        self._merge_method_summary(joint_rows, pc_results_csv)

    def _run_algorithm_trial(self, points, methods, trial_id=None):
        """Run a single deterministic or stochastic fitting pass for each method."""
        fitting_results = []
        metric_results = []
        for method, func, kwargs in methods:
            fitting_dict = {
                "method": method,
                "n_est": None,
                "d_est": None,
                "inliers": [],
                "residuals": np.array([]),
                "cpu_time": 0.0
            }
            try:
                start_time = time.time()
                result_tuple = func(points, **kwargs)
                n_est, d_est, inliers = result_tuple[:3]

                cpu_time = time.time() - start_time

                if n_est is not None and len(points) > 0:
                    n_norm = np.linalg.norm(n_est)
                    if n_norm < 1e-10:
                        residuals = np.zeros(len(points))
                    else:
                        residuals = np.abs(np.dot(points, n_est) + d_est) / n_norm
                    fitting_dict["residuals"] = residuals

                fitting_dict["n_est"] = n_est
                fitting_dict["d_est"] = d_est
                fitting_dict["inliers"] = inliers if inliers is not None else []
                fitting_dict["cpu_time"] = cpu_time

                inliers = fitting_dict["inliers"]
                if isinstance(inliers, np.ndarray) and inliers.dtype == bool:
                    inlier_count = int(np.sum(inliers))
                elif isinstance(inliers, list) and len(inliers) > 0 and isinstance(inliers[0], (bool, np.bool_)):
                    inlier_count = int(np.sum(inliers))
                elif isinstance(inliers, (list, np.ndarray)):
                    inlier_count = len(inliers)
                else:
                    inlier_count = 0
                total_points = len(points)
                inlier_ratio = inlier_count / total_points if total_points > 0 else 0
                mean_residual = np.nan

                residuals = fitting_dict["residuals"]
                if inlier_count > 0 and len(residuals) > 0:
                    if isinstance(inliers, np.ndarray) and inliers.dtype == bool:
                        inlier_residuals = residuals[inliers]
                    elif isinstance(inliers, list) and len(inliers) > 0 and isinstance(inliers[0], int):
                        inlier_residuals = residuals[inliers]
                    else:
                        inlier_residuals = []
                    mean_residual = np.mean(inlier_residuals) if len(inlier_residuals) > 0 else np.nan

                metric_results.append({
                    "Method": method,
                    "Number of Inliers": inlier_count,
                    "Inlier Ratio": inlier_ratio,
                    "Mean Inlier Residual (m)": mean_residual,
                    "CPU Time (s)": cpu_time,
                    "Normal Vector (nx,ny,nz)": f"{n_est[0]},{n_est[1]},{n_est[2]}" if n_est is not None else "Failed"
                })

            except Exception as e:
                if trial_id is not None:
                    print(f"Warning: {method} trial {trial_id + 1} failed: {e}")
                else:
                    print(f"Warning: {method} failed: {e}")
                metric_results.append({
                    "Method": method,
                    "Number of Inliers": 0,
                    "Inlier Ratio": "Failed",
                    "Mean Inlier Residual (m)": np.nan,
                    "CPU Time (s)": np.nan,
                    "Normal Vector (nx,ny,nz)": "Failed"
                })
            fitting_results.append(fitting_dict)
        return fitting_results, metric_results

    def run_deterministic_once(self, points, deterministic_methods):
        """Run each deterministic method once."""
        return self._run_algorithm_trial(points, deterministic_methods, trial_id=None)

    def run_stochastic_trial(self, points, stochastic_methods, trial_id, tau):
        """Run one stochastic trial, injecting the current RANSAC threshold when needed."""
        methods_with_tau = []
        for name, func, kwargs in stochastic_methods:
            if name == "RANSAC":
                methods_with_tau.append((name, func, {**kwargs, "tau": tau}))
            else:
                methods_with_tau.append((name, func, kwargs))
        return self._run_algorithm_trial(points, methods_with_tau, trial_id=trial_id)

    def _build_pc_results(self, all_fitting, all_results, labels, file_name, outlier_ratio):
        """Assemble per-method point-cloud summary rows for CSV export."""
        pc_results = []
        for i, res in enumerate(all_results):
            fr = all_fitting[i]
            inliers = fr.get("inliers", [])
            metrics = calculate_detection_metrics(labels, inliers)
            deviation_angle = calculate_angle_between_vectors(fr.get("n_est"), self.initial_plane_normal)
            d_diff = calculate_d_est_difference(fr.get("d_est", 0.0),self.initial_plane_interception)
            n_est_str = array_to_csv_string(fr.get("n_est"))
            residuals_str = array_to_csv_string(fr.get("residuals", np.array([])))
            inliers_str = array_to_csv_string(fr.get("inliers", []))

            pc_results.append({
                **res, **metrics,
                "Outlier Ratio": outlier_ratio,
                "Outlier Ratio (%)": outlier_ratio * 100,
                "File Name": file_name,
                "n_est": n_est_str,
                "d_est": float(fr.get("d_est", 0.0)),
                "residuals": residuals_str,
                "inliers": inliers_str,
                "Normal Deviation (deg)": deviation_angle,
                "Interception Difference (m)":d_diff
            })
        return pc_results

    def _load_fitting_from_cache(self, csv_path):
        """Load fitting results from the cached CSV file."""
        fitting_results = []
        for row in read_rows_from_csv(csv_path):
            inliers = parse_array_from_string(row.get("inliers"), dtype=bool)
            if inliers is None:
                inliers = parse_array_from_string(row.get("inliers"), dtype=int)
            fitting_results.append({
                "method": row.get("Method"),
                "n_est": parse_array_from_string(row.get("n_est")),
                "d_est": to_float(row.get("d_est")),
                "residuals": parse_array_from_string(row.get("residuals")),
                "inliers": inliers
            })
        return fitting_results

    def analyze_point_cloud(self, file_name, outlier_ratio, eps, point_cloud_dir, output_root, force_reprocess=False,
                            dpi=2000):
        """Pure-stdlib point-cloud analysis pipeline returning a list of result rows."""
        try:
            point_cloud_name = os.path.splitext(file_name)[0]
            pc_root_dir = os.path.join(output_root, point_cloud_name)
            os.makedirs(pc_root_dir, exist_ok=True)
            pc_method_plots = os.path.join(pc_root_dir, "method_plots")
            pc_comparison_plots = os.path.join(pc_root_dir, "comparison_plots")
            for dir_path in [pc_method_plots, pc_comparison_plots]:
                os.makedirs(dir_path, exist_ok=True)

            pc_results_csv = os.path.join(pc_root_dir, f"{point_cloud_name}_results.csv")
            use_cache = os.path.exists(pc_results_csv) and not force_reprocess

            file_path = os.path.join(point_cloud_dir, file_name)
            points, labels = load_point_cloud(file_path)

            if use_cache:
                print(f"Using cached results: {pc_results_csv}")
                fitting_results = self._load_fitting_from_cache(pc_results_csv)
            else:
                print(f"Processing {file_name}: points={len(points)}, outlier_ratio={outlier_ratio * 100:.1f}%")
                algo_categories = self._get_algorithm_categories(eps)
                stochastic_methods = algo_categories["stochastic_methods"]
                deterministic_methods = algo_categories["deterministic_methods"]

                det_fitting, det_results = self.run_deterministic_once(points, deterministic_methods)
                sto_fitting, sto_results = self.run_stochastic_trial(
                    points, stochastic_methods, trial_id=1, tau=self.tau
                )
                all_fitting = sto_fitting + det_fitting
                all_results = sto_results + det_results

                pc_results = self._build_pc_results(all_fitting, all_results, labels, file_name, outlier_ratio)
                write_rows_to_csv(pc_results_csv, pc_results)
                print(f"Saved point-cloud results to: {pc_results_csv}")
                fitting_results = all_fitting

            cached_rows = read_rows_from_csv(pc_results_csv)
            if len(points) == 0:
                return cached_rows

            valid_fitting = [fr for fr in fitting_results if fr["n_est"] is not None and fr["d_est"] is not None]
            if not valid_fitting:
                print(f"Warning: {point_cloud_name} has no valid fitting result; skipping plot generation.")
                return cached_rows

            common_3d_path = os.path.join(pc_root_dir, f"{point_cloud_name}_3d_terrain.png")
            common_3d_fig = plot_common_3d_terrain(
                points=points,
                labels=labels,
                fitting_results=valid_fitting,
                show_plot=False,
                save_plot=force_reprocess,
                save_path=common_3d_path,
                dpi=dpi,
            )
            plt.close(common_3d_fig)

            downsample_step = 20 if len(points) > 10000 else 1
            for fr in valid_fitting:
                method = fr["method"]
                residuals = fr.get("residuals")
                if residuals is None:
                    continue

                if len(points) > 10000:
                    downsampled_points, downsampled_residuals = downsample_data(points, residuals, step=downsample_step)
                else:
                    downsampled_points, downsampled_residuals = points, residuals

                rough_path = os.path.join(pc_method_plots, f"{method}_roughness.png")
                plot_roughness_heatmap(
                    downsampled_points,
                    np.abs(downsampled_residuals),
                    method,
                    show_plot=False,
                    save_plot=force_reprocess,
                    save_path=rough_path,
                )

                elev_path = os.path.join(pc_method_plots, f"{method}_elevation.png")
                plot_elevation_profile(
                    downsampled_points,
                    fr["n_est"],
                    fr["d_est"],
                    method,
                    show_plot=False,
                    save_plot=force_reprocess,
                    save_path=elev_path,
                    bin_width=1.0,
                )
                print(f"Saved method plots for {point_cloud_name} / {method}: {rough_path}, {elev_path}")

            elev_trend_path = os.path.join(pc_comparison_plots, "2_2_elevation_trend.png")
            plot_elevation_trend_comparison(
                points,
                valid_fitting,
                show_plot=False,
                save_plot=force_reprocess,
                save_path=elev_trend_path,
                bin_width=1.0,
            )

            summary_path = os.path.join(pc_comparison_plots, "5_roughness_elevation_summary.png")
            plot_roughness_elevation_summary(
                points=points,
                fitting_results=valid_fitting,
                show_plot=False,
                save_plot=force_reprocess,
                save_path=summary_path,
            )
            print(f"Saved comparison plots for {point_cloud_name}: {elev_trend_path}, {summary_path}")

            risk_result = None
            safety_analysis = None
            if self.enable_risk_analysis:
                risk_result = self._run_risk_probability_analysis(
                    points=points,
                    labels=labels,
                    pc_root_dir=pc_root_dir,
                    point_cloud_name=point_cloud_name,
                )
            if self.enable_safety_analysis:
                safety_analysis = self._run_safety_analysis(
                    points=points,
                    fitting_results=valid_fitting,
                    pc_root_dir=pc_root_dir,
                    point_cloud_name=point_cloud_name,
                    pc_results_csv=pc_results_csv,
                )
            if self.enable_joint_analysis:
                self._run_joint_analysis(
                    risk_result=risk_result,
                    safety_analysis=safety_analysis,
                    pc_root_dir=pc_root_dir,
                    point_cloud_name=point_cloud_name,
                    pc_results_csv=pc_results_csv,
                )

            if self.export_formats:
                print(f"\n=== Exporting {point_cloud_name} results ===")
                algo_categories = self._get_algorithm_categories(eps)
                sto_names = [m[0] for m in algo_categories["stochastic_methods"]]
                for fr in valid_fitting:
                    method_name = fr["method"]
                    if method_name == "RWTLS" and fr["n_est"] is None:
                        print(f"Warning: skipping {method_name} export because it ran out of memory.")
                        continue

                    suffix = "_trial_1" if method_name in sto_names else ""
                    if "cloudcompare" in self.export_formats:
                        export_to_cloudcompare(
                            points,
                            fr["inliers"],
                            fr["n_est"],
                            fr["d_est"],
                            f"{method_name}{suffix}",
                            os.path.join(pc_root_dir, "cloudcompare"),
                        )
                    if "matlab" in self.export_formats:
                        export_to_matlab(
                            points,
                            fr["inliers"],
                            fr["n_est"],
                            fr["d_est"],
                            f"{method_name}{suffix}",
                            os.path.join(pc_root_dir, "matlab"),
                        )
                print(f"Exported files for {point_cloud_name} successfully.")

            return read_rows_from_csv(pc_results_csv)

        except Exception as e:
            print(f"Failed to process {file_name}: {e}")
            import traceback
            traceback.print_exc()
            return []

    def run_outlier_sweep(self, point_cloud_dir="point_clouds", output_root="outlier_sweep_results",
                          random_seed=42, force_reprocess=False, dpi=2000, save_vector=False,
                          eps_mapping=None, max_workers=1):
        """Pure-stdlib outlier ratio sweep."""
        np.random.seed(random_seed)
        os.makedirs(output_root, exist_ok=True)
        global_comparison_dir = os.path.join(output_root, "global_comparison_plots")
        os.makedirs(global_comparison_dir, exist_ok=True)

        if eps_mapping is None:
            eps_mapping = _identity_eps_mapping

        pc_files = [f for f in os.listdir(point_cloud_dir) if f.endswith((".txt", ".xyz"))]
        pc_files_with_ratio = []
        for file in pc_files:
            try:
                ratio = extract_outlier_ratio(file)
                pc_files_with_ratio.append((file, ratio))
            except ValueError as e:
                print(f"Warning: skipping file {file}: {e}")
        pc_files_with_ratio.sort(key=lambda item: item[1])

        if not pc_files_with_ratio:
            print("Warning: no valid point-cloud files were found.")
            return []

        sample_eps = eps_mapping(pc_files_with_ratio[0][1])
        algo_categories = self._get_algorithm_categories(sample_eps)
        all_methods = algo_categories["all_methods"]
        max_workers = min(_resolve_max_workers(max_workers), len(pc_files_with_ratio))

        print(f"\n===== Processing all point clouds (count={len(pc_files_with_ratio)}) =====")
        tasks = []
        for file_name, outlier_ratio in pc_files_with_ratio:
            tasks.append({
                "file_name": file_name,
                "outlier_ratio": outlier_ratio,
                "eps": eps_mapping(outlier_ratio),
                "point_cloud_dir": point_cloud_dir,
                "output_root": output_root,
                "force_reprocess": force_reprocess,
                "dpi": dpi,
            })

        results = []
        if max_workers <= 1:
            for task in tqdm(tasks, desc="Total progress"):
                result_rows = self.analyze_point_cloud(**task)
                results.append(result_rows)
        else:
            analyzer_kwargs = self._build_analyzer_kwargs()
            executor, backend = _build_parallel_executor(max_workers)
            print(f"Using {backend}-based parallel execution with {max_workers} workers.")
            with executor:
                future_to_task = {
                    executor.submit(_analyze_point_cloud_worker, analyzer_kwargs, task): task
                    for task in tasks
                }
                progress = tqdm(total=len(tasks), desc="Total progress")
                try:
                    for future in as_completed(future_to_task):
                        task = future_to_task[future]
                        try:
                            result_rows = future.result()
                        except Exception as exc:
                            print(f"Failed to process {task['file_name']} in worker: {exc}")
                            result_rows = []
                        results.append(result_rows)
                        progress.update(1)
                finally:
                    progress.close()

        global_results_list = []
        for result_rows in results:
            if result_rows:
                global_results_list.extend(result_rows)

        if not global_results_list:
            print("\nWarning: no global result rows were produced, so comparison plots were skipped.")
            return []

        method_order = {method: index for index, method in enumerate(all_methods)}
        global_agg_rows = sorted(
            global_results_list,
            key=lambda row: (
                to_float(row.get("Outlier Ratio"), default=np.inf),
                method_order.get(row.get("Method"), len(all_methods)),
            ),
        )
        global_agg_csv = os.path.join(global_comparison_dir, "global_aggregated_results.csv")
        write_rows_to_csv(global_agg_csv, global_agg_rows)
        print(f"\nSaved global aggregated results to: {global_agg_csv}")

        self.generate_outlier_ratio_plots(global_agg_rows, all_methods, global_comparison_dir, dpi, save_vector)

        global_risk_rows = self._collect_global_risk_summaries(pc_files_with_ratio, output_root)
        if global_risk_rows:
            global_risk_rows = sorted(
                global_risk_rows,
                key=lambda row: to_float(row.get("Outlier Ratio"), default=np.inf),
            )
            global_risk_csv = os.path.join(global_comparison_dir, "global_risk_results.csv")
            write_rows_to_csv(global_risk_csv, global_risk_rows)
            print(f"Saved global risk summary to: {global_risk_csv}")
            self.generate_global_risk_probability_plots(
                global_risk_rows,
                global_comparison_dir,
                dpi=dpi,
                save_vector=save_vector,
            )
        return global_agg_rows

    def generate_outlier_ratio_plots(self, agg_df, all_methods, comparison_dir, dpi=1200, save_vector=False):
        """Generate outlier-ratio trend plots using plain row records."""
        print("\n=== Generating outlier ratio performance plots ===")
        numeric_cols = [
            "Accuracy",
            "False Negative Rate",
            "False Positive Rate",
            "Mean Inlier Residual (m)",
            "CPU Time (s)",
            "Normal Deviation (deg)",
            "Interception Difference (m)",
            "Mean Safety Coefficient",
            "Best Window Mean Safety",
            "Safety Map Abs Difference Sum",
            "Mean Joint Hazard",
            "High Joint Hazard Ratio",
            "Mean Joint Safe Score",
            "Best Joint Window Score",
            "Joint Safe Map Abs Difference Sum",
            "Joint Map Accuracy",
            "Joint Map Precision",
            "Joint Map Recall",
            "Joint Map F1",
        ]
        available_cols = set()
        for row in agg_df:
            available_cols.update(row.keys())
        numeric_cols = [col for col in numeric_cols if col in available_cols]

        for metric in numeric_cols:
            try:
                x_min_inset, x_max_inset = 10, 40
                inset_series_by_method = {}
                all_metric_values = []
                plotted_methods = []
                method_handles = []

                with plt.rc_context({
                    "font.family": ["Times New Roman", "DejaVu Serif", "serif"],
                    "axes.unicode_minus": True,
                    "axes.labelsize": 11,
                    "axes.titlesize": 10,
                    "legend.fontsize": 9,
                    "xtick.labelsize": 9,
                    "ytick.labelsize": 9,
                }):
                    fig = plt.figure(figsize=(10.6, 6.6))
                    grid = fig.add_gridspec(
                        2,
                        1,
                        height_ratios=[2.4, 1.15],
                        hspace=0.14,
                    )
                    ax_main = fig.add_subplot(grid[0, 0])
                    ax_zoom = fig.add_subplot(grid[1, 0])

                    for method in all_methods:
                        metric_pairs = _extract_metric_pairs(agg_df, metric, method=method)
                        if not metric_pairs:
                            continue

                        x_values = [pair[0] for pair in metric_pairs]
                        y_values = [pair[1] for pair in metric_pairs]
                        inset_pairs = [
                            pair for pair in metric_pairs
                            if x_min_inset <= pair[0] <= x_max_inset
                        ]
                        inset_series_by_method[method] = inset_pairs
                        all_metric_values.extend(y_values)
                        plotted_methods.append(method)

                        line_style = dict(
                            color=ALGORITHM_COLORS.get(method, "#333333"),
                            linewidth=1.8,
                            marker=ALGORITHM_MARKERS.get(method, "o"),
                            markersize=4.2,
                            markeredgecolor="white",
                            markeredgewidth=0.45,
                            markevery=max(1, len(x_values) // 10),
                        )
                        line = ax_main.plot(x_values, y_values, label=method, **line_style)[0]
                        if inset_pairs:
                            inset_x = [pair[0] for pair in inset_pairs]
                            inset_y = [pair[1] for pair in inset_pairs]
                            ax_zoom.plot(inset_x, inset_y, **line_style)
                        method_handles.append(line)

                    if not plotted_methods:
                        plt.close(fig)
                        continue

                    use_log_scale = _should_use_log_scale(metric, all_metric_values)
                    _style_journal_axis(ax_main, metric, use_log_scale=use_log_scale)
                    _style_journal_axis(ax_zoom, metric, use_log_scale=use_log_scale)

                    ax_main.set_ylabel(metric)
                    ax_main.tick_params(labelbottom=False)
                    ax_zoom.set_xlabel("Outlier Ratio (%)")
                    ax_zoom.set_ylabel(metric)

                    full_x_values = sorted({pair[0] for method in plotted_methods for pair in _extract_metric_pairs(agg_df, metric, method=method)})
                    if full_x_values:
                        ax_main.set_xlim(min(full_x_values), max(full_x_values))
                    ax_zoom.set_xlim(x_min_inset, x_max_inset)

                    if use_log_scale:
                        positive_all = [value for value in all_metric_values if value > 0]
                        positive_inset = [
                            pair[1]
                            for method in plotted_methods
                            for pair in inset_series_by_method.get(method, [])
                            if pair[1] > 0
                        ]
                        if positive_all:
                            ax_main.set_ylim(min(positive_all) * 0.88, max(positive_all) * 1.15)
                        if positive_inset:
                            ax_zoom.set_ylim(min(positive_inset) * 0.9, max(positive_inset) * 1.12)
                    else:
                        full_limits = _compute_linear_limits(
                            all_metric_values,
                            bounded=_metric_is_bounded(metric),
                        )
                        inset_values = [
                            pair[1]
                            for method in plotted_methods
                            for pair in inset_series_by_method.get(method, [])
                        ]
                        inset_limits = _compute_linear_limits(
                            inset_values if inset_values else all_metric_values,
                            bounded=_metric_is_bounded(metric),
                        )
                        if full_limits:
                            ax_main.set_ylim(*full_limits)
                        if inset_limits:
                            ax_zoom.set_ylim(*inset_limits)

                    ax_main.set_title("(a) Full outlier-ratio range", loc="left", fontweight="bold", pad=6)
                    ax_zoom.set_title(
                        f"(b) Low outlier-ratio regime ({x_min_inset}%–{x_max_inset}%)",
                        loc="left",
                        fontweight="bold",
                        pad=6,
                    )

                    summary_rows = []
                    for method in plotted_methods:
                        metric_pairs = _extract_metric_pairs(agg_df, metric, method=method)
                        full_mean = mean_ignore_nan([pair[1] for pair in metric_pairs])
                        inset_mean = mean_ignore_nan([pair[1] for pair in inset_series_by_method.get(method, [])])
                        summary_rows.append((method, full_mean, inset_mean))

                    summary_text = ["Mean values", "10%-90%    10%-40%"]
                    for method, full_mean, inset_mean in summary_rows:
                        full_text = f"{full_mean:.3f}" if not np.isnan(full_mean) else "nan"
                        inset_text = f"{inset_mean:.3f}" if not np.isnan(inset_mean) else "nan"
                        summary_text.append(f"{method:<12}{full_text:>7}    {inset_text:>7}")

                    ax_main.text(
                        1.01,
                        0.98,
                        "\n".join(summary_text),
                        transform=ax_main.transAxes,
                        va="top",
                        ha="left",
                        fontsize=8.3,
                        family="monospace",
                        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fbfbfb", edgecolor="#d0d0d0", alpha=0.96),
                    )

                    legend_columns = 3 if len(method_handles) <= 6 else 4
                    fig.legend(
                        method_handles,
                        plotted_methods,
                        loc="upper center",
                        bbox_to_anchor=(0.5, 0.995),
                        ncol=legend_columns,
                        frameon=False,
                        handlelength=2.2,
                        columnspacing=1.2,
                    )

                    fig.subplots_adjust(left=0.10, right=0.77, top=0.86, bottom=0.10)
                    os.makedirs(comparison_dir, exist_ok=True)
                    trend_path_png = os.path.join(comparison_dir, f"outlier_vs_{_sanitize_metric_name(metric)}.png")
                    fig.savefig(trend_path_png, dpi=dpi, bbox_inches="tight")
                    if save_vector:
                        trend_path_pdf = os.path.join(comparison_dir, f"outlier_vs_{_sanitize_metric_name(metric)}.pdf")
                        fig.savefig(trend_path_pdf, format="pdf", bbox_inches="tight")

                    plt.close(fig)
                    print(f"Saved: {trend_path_png}")

            except Exception as e:
                print(f"Failed to generate {metric} plot: {e}")


def array_to_csv_string(arr, use_scientific=True):
    """Serialize an array-like object to a single-line CSV-style string."""
    if arr is None or (hasattr(arr, "__len__") and len(arr) == 0):
        return "[]"

    arr = np.asarray(arr)

    # Keep arrays on one line so they can round-trip through CSV caches.
    opts = {
        "threshold": np.inf,
        "linewidth": np.inf,
        "precision": 8,
    }
    if not use_scientific:
        opts["suppress"] = True

    with np.printoptions(**opts):
        s = np.array2string(arr, separator=", ")
        return s.replace("\n", " ").replace("  ", " ")
