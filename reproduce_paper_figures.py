import argparse
import math
import os

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from constants import ALGORITHM_COLORS, ALGORITHM_MARKERS, PLOT_CONFIG
from experiment_presets import get_paper_preset
from tabular_results import (
    mean_ignore_nan,
    read_rows_from_csv,
    std_ignore_nan,
    to_float,
    write_rows_to_csv,
)


RISK_METRICS = {
    "Mean Risk Probability": ["Mean Risk Probability"],
    "High Risk Ratio": ["High Risk Ratio"],
    "Risk Map Accuracy": ["Risk Map Accuracy"],
    "Risk Map F1": ["Risk Map F1"],
}

FITTING_METRICS = {
    "Accuracy": ["Accuracy"],
    "Mean Inlier Residual (m)": ["Mean Inlier Residual (m)"],
    "CPU Time (s)": ["CPU Time (s)"],
    "Normal Deviation": ["Normal Deviation (deg)", "Normal Deviation (°)", "Normal Deviation (掳)"],
    "Interception Difference (m)": ["Interception Difference (m)"],
    "Inlier Ratio": ["Inlier Ratio"],
}

SAFETY_JOINT_METRICS = {
    "Mean Safety Coefficient": ["Mean Safety Coefficient"],
    "Best Window Mean Safety": ["Best Window Mean Safety"],
    "Mean Joint Safe Score": ["Mean Joint Safe Score"],
    "Best Joint Window Score": ["Best Joint Window Score"],
    "Joint Map F1": ["Joint Map F1"],
    "Mean Joint Hazard": ["Mean Joint Hazard"],
}

METHOD_RANKING_OBJECTIVES = {
    "Accuracy": "max",
    "Mean Inlier Residual (m)": "min",
    "CPU Time (s)": "min",
    "Normal Deviation": "min",
    "Interception Difference (m)": "min",
    "Mean Safety Coefficient": "max",
    "Best Window Mean Safety": "max",
    "Mean Joint Safe Score": "max",
    "Best Joint Window Score": "max",
    "Joint Map F1": "max",
    "Mean Joint Hazard": "min",
}

PRESET_RESULT_FALLBACKS = {
    "peiv10": [
        "results/outlier_sweep_results_peiv",
        "results/outlier_sweep_results",
    ],
    "peiv15": [
        "results/outlier_sweep_results_peiv15",
    ],
    "peiv35": [
        "results/outlier_sweep_results_peiv35",
    ],
    "structured35": [
        "results/paper_structured35",
        "results/outlier_sweep_results_structured35",
    ],
}


def sanitize_name(name):
    return name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")


def ordered_unique(values):
    seen = set()
    ordered = []
    for value in values:
        if value in seen or value in {None, ""}:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def resolve_metric_name(rows, aliases):
    available = set()
    for row in rows:
        available.update(row.keys())
    for alias in aliases:
        if alias in available:
            return alias
    return None


def extract_metric_pairs(rows, metric_name, method=None):
    pairs = []
    for row in rows:
        if method is not None and row.get("Method") != method:
            continue
        x_value = to_float(row.get("Outlier Ratio (%)"))
        y_value = to_float(row.get(metric_name))
        if np.isnan(x_value) or np.isnan(y_value):
            continue
        pairs.append((x_value, y_value))
    pairs.sort(key=lambda item: item[0])
    return pairs


def resolve_results_root(preset_name=None, results_root=None):
    if results_root:
        return os.path.abspath(results_root)
    if preset_name is None:
        raise ValueError("Either preset_name or results_root must be provided.")

    preset = get_paper_preset(preset_name)
    candidates = [preset.output_root]
    candidates.extend(PRESET_RESULT_FALLBACKS.get(preset_name, []))
    candidates.append(os.path.join("results", f"outlier_sweep_results_{preset_name}"))

    for candidate in candidates:
        abs_candidate = os.path.abspath(candidate)
        agg_csv = os.path.join(abs_candidate, "global_comparison_plots", "global_aggregated_results.csv")
        if os.path.exists(agg_csv):
            return abs_candidate
    return os.path.abspath(preset.output_root)


def load_result_rows(results_root):
    global_dir = os.path.join(results_root, "global_comparison_plots")
    agg_csv = os.path.join(global_dir, "global_aggregated_results.csv")
    risk_csv = os.path.join(global_dir, "global_risk_results.csv")

    agg_rows = read_rows_from_csv(agg_csv)
    if not agg_rows:
        raise FileNotFoundError(f"Missing aggregated result CSV: {agg_csv}")

    risk_rows = read_rows_from_csv(risk_csv)
    return agg_rows, risk_rows, global_dir


def save_figure(fig, output_path, save_vector=False):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if save_vector:
        vector_path = os.path.splitext(output_path)[0] + ".pdf"
        fig.savefig(vector_path, format="pdf", bbox_inches="tight")


def build_metric_summary(rows, metric_aliases):
    summary_rows = []
    for label, aliases in metric_aliases.items():
        metric_name = resolve_metric_name(rows, aliases)
        if metric_name is None:
            continue
        values = [to_float(row.get(metric_name)) for row in rows]
        values = [value for value in values if not np.isnan(value)]
        if not values:
            continue
        summary_rows.append({
            "Metric": label,
            "Resolved Metric": metric_name,
            "Count": len(values),
            "Mean": mean_ignore_nan(values),
            "Std": std_ignore_nan(values),
            "Min": float(np.min(values)),
            "Max": float(np.max(values)),
        })
    return summary_rows


def build_method_summary(rows, methods, metric_aliases):
    summary_rows = []
    for method in methods:
        method_rows = [row for row in rows if row.get("Method") == method]
        if not method_rows:
            continue
        summary_row = {"Method": method}
        for label, aliases in metric_aliases.items():
            metric_name = resolve_metric_name(method_rows, aliases)
            if metric_name is None:
                continue
            values = [to_float(row.get(metric_name)) for row in method_rows]
            values = [value for value in values if not np.isnan(value)]
            if not values:
                continue
            summary_row[f"{label} Mean"] = mean_ignore_nan(values)
            summary_row[f"{label} Std"] = std_ignore_nan(values)
        summary_rows.append(summary_row)
    return summary_rows


def build_method_rankings(rows, methods):
    ranking_rows = []
    for label, objective in METHOD_RANKING_OBJECTIVES.items():
        aliases = FITTING_METRICS.get(label) or SAFETY_JOINT_METRICS.get(label)
        if aliases is None:
            continue
        best_method = None
        best_value = np.nan
        for method in methods:
            method_rows = [row for row in rows if row.get("Method") == method]
            metric_name = resolve_metric_name(method_rows, aliases)
            if metric_name is None:
                continue
            values = [to_float(row.get(metric_name)) for row in method_rows]
            values = [value for value in values if not np.isnan(value)]
            if not values:
                continue
            current_value = mean_ignore_nan(values)
            if np.isnan(current_value):
                continue
            if best_method is None:
                best_method = method
                best_value = current_value
                continue
            if objective == "max" and current_value > best_value:
                best_method = method
                best_value = current_value
            if objective == "min" and current_value < best_value:
                best_method = method
                best_value = current_value
        if best_method is not None:
            ranking_rows.append({
                "Metric": label,
                "Objective": objective,
                "Best Method": best_method,
                "Best Mean Value": best_value,
            })
    return ranking_rows


def plot_risk_grid(risk_rows, metric_aliases, output_path, title, save_vector=False):
    resolved_metrics = []
    for label, aliases in metric_aliases.items():
        metric_name = resolve_metric_name(risk_rows, aliases)
        if metric_name is not None:
            resolved_metrics.append((label, metric_name))
    if not resolved_metrics:
        return None

    ncols = 2
    nrows = math.ceil(len(resolved_metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.8 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (label, metric_name) in zip(axes, resolved_metrics):
        pairs = extract_metric_pairs(risk_rows, metric_name)
        if not pairs:
            ax.set_visible(False)
            continue
        x_values = [pair[0] for pair in pairs]
        y_values = [pair[1] for pair in pairs]
        ax.plot(x_values, y_values, color="#bc4749", marker="o", linewidth=1.8, markersize=4)
        ax.set_title(label, fontsize=PLOT_CONFIG["font_size"])
        ax.set_xlabel("Outlier Ratio (%)", fontsize=PLOT_CONFIG["font_size"] * 0.9)
        ax.set_ylabel(label, fontsize=PLOT_CONFIG["font_size"] * 0.9)
        ax.grid(linestyle=PLOT_CONFIG["grid_style"], alpha=PLOT_CONFIG["grid_alpha"])

    for ax in axes[len(resolved_metrics):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=PLOT_CONFIG["title_size"])
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, output_path, save_vector=save_vector)
    plt.close(fig)
    return output_path


def plot_method_grid(rows, methods, metric_aliases, output_path, title, save_vector=False):
    resolved_metrics = []
    for label, aliases in metric_aliases.items():
        metric_name = resolve_metric_name(rows, aliases)
        if metric_name is not None:
            resolved_metrics.append((label, metric_name))
    if not resolved_metrics:
        return None

    ncols = 2
    nrows = math.ceil(len(resolved_metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8 * ncols, 4.9 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (label, metric_name) in zip(axes, resolved_metrics):
        any_series = False
        for method in methods:
            pairs = extract_metric_pairs(rows, metric_name, method=method)
            if not pairs:
                continue
            any_series = True
            x_values = [pair[0] for pair in pairs]
            y_values = [pair[1] for pair in pairs]
            ax.plot(
                x_values,
                y_values,
                color=ALGORITHM_COLORS.get(method, "#000000"),
                marker=ALGORITHM_MARKERS.get(method, "o"),
                linewidth=1.5,
                markersize=4,
                label=method,
            )
        if not any_series:
            ax.set_visible(False)
            continue
        ax.set_title(label, fontsize=PLOT_CONFIG["font_size"])
        ax.set_xlabel("Outlier Ratio (%)", fontsize=PLOT_CONFIG["font_size"] * 0.9)
        ax.set_ylabel(label, fontsize=PLOT_CONFIG["font_size"] * 0.9)
        ax.grid(linestyle=PLOT_CONFIG["grid_style"], alpha=PLOT_CONFIG["grid_alpha"])

    for ax in axes[len(resolved_metrics):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(title, fontsize=PLOT_CONFIG["title_size"])
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, output_path, save_vector=save_vector)
    plt.close(fig)
    return output_path


def write_summary_markdown(output_path, results_root, methods, ranking_rows, generated_files):
    lines = [
        "# Paper-Style Reproduction Summary",
        "",
        f"- Results root: `{results_root}`",
        f"- Methods covered: {', '.join(methods)}",
        "",
        "## Best Methods by Metric",
        "",
    ]
    for row in ranking_rows:
        lines.append(
            f"- {row['Metric']}: `{row['Best Method']}` ({row['Objective']}, mean={row['Best Mean Value']:.4f})"
        )
    lines.append("")
    lines.append("## Generated Artifacts")
    lines.append("")
    for label, path in generated_files.items():
        lines.append(f"- {label}: `{path}`")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def reproduce_paper_artifacts(preset_name=None, results_root=None, output_dir=None, save_vector=False):
    resolved_results_root = resolve_results_root(preset_name=preset_name, results_root=results_root)
    agg_rows, risk_rows, _ = load_result_rows(resolved_results_root)
    methods = ordered_unique([row.get("Method") for row in agg_rows])

    if output_dir is None:
        output_dir = os.path.join(resolved_results_root, "paper_reproduction")
    output_dir = os.path.abspath(output_dir)

    chapter2_dir = os.path.join(output_dir, "chapter2_macro_risk")
    chapter3_fitting_dir = os.path.join(output_dir, "chapter3_fitting")
    chapter3_safety_dir = os.path.join(output_dir, "chapter3_safety_joint")
    tables_dir = os.path.join(output_dir, "tables")

    generated_files = {}

    risk_plot = plot_risk_grid(
        risk_rows,
        RISK_METRICS,
        os.path.join(chapter2_dir, "figure_2_macro_risk_trends.png"),
        "Chapter 2: Macro Risk Probability Trends",
        save_vector=save_vector,
    )
    if risk_plot:
        generated_files["Chapter 2 Risk Figure"] = risk_plot

    fitting_plot = plot_method_grid(
        agg_rows,
        methods,
        FITTING_METRICS,
        os.path.join(chapter3_fitting_dir, "figure_3_fitting_metrics.png"),
        "Chapter 3: Plane Fitting Metrics",
        save_vector=save_vector,
    )
    if fitting_plot:
        generated_files["Chapter 3 Fitting Figure"] = fitting_plot

    safety_plot = plot_method_grid(
        agg_rows,
        methods,
        SAFETY_JOINT_METRICS,
        os.path.join(chapter3_safety_dir, "figure_4_safety_joint_metrics.png"),
        "Chapter 3: Safety and Joint Decision Metrics",
        save_vector=save_vector,
    )
    if safety_plot:
        generated_files["Chapter 3 Safety Figure"] = safety_plot

    risk_table = build_metric_summary(risk_rows, RISK_METRICS)
    fitting_table = build_method_summary(agg_rows, methods, FITTING_METRICS)
    safety_table = build_method_summary(agg_rows, methods, SAFETY_JOINT_METRICS)
    ranking_table = build_method_rankings(agg_rows, methods)

    risk_table_path = os.path.join(tables_dir, "table_2_macro_risk_summary.csv")
    fitting_table_path = os.path.join(tables_dir, "table_3_fitting_method_summary.csv")
    safety_table_path = os.path.join(tables_dir, "table_4_safety_joint_method_summary.csv")
    ranking_table_path = os.path.join(tables_dir, "table_method_rankings.csv")
    write_rows_to_csv(risk_table_path, risk_table)
    write_rows_to_csv(fitting_table_path, fitting_table)
    write_rows_to_csv(safety_table_path, safety_table)
    write_rows_to_csv(ranking_table_path, ranking_table)

    generated_files["Risk Summary Table"] = risk_table_path
    generated_files["Fitting Summary Table"] = fitting_table_path
    generated_files["Safety Summary Table"] = safety_table_path
    generated_files["Method Ranking Table"] = ranking_table_path

    summary_md = os.path.join(output_dir, "paper_style_summary.md")
    write_summary_markdown(summary_md, resolved_results_root, methods, ranking_table, generated_files)
    generated_files["Markdown Summary"] = summary_md

    return {
        "results_root": resolved_results_root,
        "output_dir": output_dir,
        "files": generated_files,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce paper-style figures and tables from experiment outputs.")
    parser.add_argument("--preset", default="peiv35", help="Preset name used to infer the result directory.")
    parser.add_argument("--results-root", default=None, help="Explicit experiment result root.")
    parser.add_argument("--output-dir", default=None, help="Directory for reproduced figures and tables.")
    parser.add_argument("--save-vector", action="store_true", help="Also save PDF figures.")
    return parser.parse_args()


def main():
    args = parse_args()
    report = reproduce_paper_artifacts(
        preset_name=args.preset,
        results_root=args.results_root,
        output_dir=args.output_dir,
        save_vector=args.save_vector,
    )
    print(f"Paper-style reproduction saved to: {report['output_dir']}")


if __name__ == "__main__":
    main()
