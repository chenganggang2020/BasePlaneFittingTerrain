import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import zoom

from risk_probability_map import evaluate_risk_map_against_labels


def _align_map_to_shape(map_data, target_shape):
    map_data = np.asarray(map_data, dtype=float)
    if map_data.shape == target_shape:
        return map_data
    zoom_factors = (
        target_shape[0] / map_data.shape[0],
        target_shape[1] / map_data.shape[1],
    )
    return zoom(map_data, zoom_factors, order=1)


def combine_risk_and_safety(risk_probability, safety_coefficient, risk_weight=0.55, safety_weight=0.45):
    """Fuse macro risk and micro safety into joint hazard and joint safe-score maps."""
    total_weight = max(risk_weight + safety_weight, 1e-12)
    risk_weight = risk_weight / total_weight
    safety_weight = safety_weight / total_weight

    risk_probability = np.clip(np.asarray(risk_probability, dtype=float), 0.0, 1.0)
    safety_coefficient = np.clip(np.asarray(safety_coefficient, dtype=float), 0.0, 1.0)

    joint_hazard = np.clip(
        risk_weight * risk_probability + safety_weight * (1.0 - safety_coefficient),
        0.0,
        1.0,
    )
    joint_safe_score = np.clip(
        ((1.0 - risk_probability) ** risk_weight) * (np.maximum(safety_coefficient, 0.0) ** safety_weight),
        0.0,
        1.0,
    )
    return joint_hazard, joint_safe_score


def _rename_metrics(metrics, prefix):
    renamed = {}
    for key, value in metrics.items():
        if key.startswith("Risk Map "):
            renamed[key.replace("Risk Map", prefix, 1)] = value
        else:
            renamed[f"{prefix} {key}"] = value
    return renamed


def search_joint_safe_windows(joint_safe_score, x_grid, y_grid, window_size=5, min_weight=0.35):
    """Search joint windows with a soft score to avoid all-zero degeneracy."""
    joint_safe_score = np.asarray(joint_safe_score, dtype=float)
    rows, cols = joint_safe_score.shape

    if isinstance(window_size, int):
        window_rows = window_cols = window_size
    else:
        window_rows, window_cols = window_size

    if rows < window_rows or cols < window_cols:
        raise ValueError(
            f"Window size {(window_rows, window_cols)} is larger than joint score grid {joint_safe_score.shape}"
        )

    mean_weight = 1.0 - min_weight
    score_rows = rows - window_rows + 1
    score_cols = cols - window_cols + 1
    score_map = np.zeros((score_rows, score_cols), dtype=float)

    best_score = -np.inf
    best_window = None
    for row in range(score_rows):
        for col in range(score_cols):
            window = joint_safe_score[row : row + window_rows, col : col + window_cols]
            mean_score = float(np.mean(window))
            min_score = float(np.min(window))
            window_score = mean_weight * mean_score + min_weight * min_score
            score_map[row, col] = window_score

            if window_score > best_score:
                center_row = row + window_rows // 2
                center_col = col + window_cols // 2
                best_score = window_score
                best_window = {
                    "top_left_row": row,
                    "top_left_col": col,
                    "center_row": center_row,
                    "center_col": center_col,
                    "center_x": float(x_grid[center_row, center_col]),
                    "center_y": float(y_grid[center_row, center_col]),
                    "mean_score": mean_score,
                    "min_score": min_score,
                    "window_score": window_score,
                    "window_rows": window_rows,
                    "window_cols": window_cols,
                }

    return {
        "score_map": score_map,
        "best_window": best_window,
    }


def evaluate_joint_risk_safety(
    risk_result,
    safety_result,
    risk_weight=0.55,
    safety_weight=0.45,
    label_grid=None,
):
    """Evaluate a joint decision map from risk probability and safety coefficient."""
    safety_map = np.asarray(safety_result["safety_coefficient"], dtype=float)
    x_grid = safety_result["x_grid"]
    y_grid = safety_result["y_grid"]
    target_shape = safety_map.shape

    risk_map = _align_map_to_shape(risk_result["risk_probability"], target_shape)
    if label_grid is None:
        label_grid = risk_result.get("label_grid")
    if label_grid is not None:
        label_grid = _align_map_to_shape(label_grid, target_shape)
        label_grid = np.rint(label_grid).astype(int)

    joint_hazard, joint_safe_score = combine_risk_and_safety(
        risk_map,
        safety_map,
        risk_weight=risk_weight,
        safety_weight=safety_weight,
    )
    window_result = search_joint_safe_windows(
        joint_safe_score,
        x_grid,
        y_grid,
        window_size=safety_result["window_size"],
    )
    metrics = (
        _rename_metrics(
            evaluate_risk_map_against_labels(joint_hazard, label_grid, threshold=0.5),
            "Joint Map",
        )
        if label_grid is not None
        else {}
    )

    return {
        "risk_probability": risk_map,
        "safety_coefficient": safety_map,
        "joint_hazard": joint_hazard,
        "joint_safe_score": joint_safe_score,
        "score_map": window_result["score_map"],
        "best_window": window_result["best_window"],
        "x_grid": x_grid,
        "y_grid": y_grid,
        "label_grid": label_grid,
        "risk_weight": risk_weight,
        "safety_weight": safety_weight,
        "mean_joint_hazard": float(np.mean(joint_hazard)),
        "max_joint_hazard": float(np.max(joint_hazard)),
        "mean_joint_safe_score": float(np.mean(joint_safe_score)),
        "max_joint_safe_score": float(np.max(joint_safe_score)),
        "high_joint_hazard_ratio": float(np.mean(joint_hazard >= 0.5)),
        "metrics": metrics,
    }


def compare_joint_safe_maps(reference_result, candidate_result):
    """Compare two joint safe-score maps by absolute per-cell difference."""
    reference = np.asarray(reference_result["joint_safe_score"], dtype=float)
    candidate = np.asarray(candidate_result["joint_safe_score"], dtype=float)
    if reference.shape != candidate.shape:
        candidate = _align_map_to_shape(candidate, reference.shape)
    return float(np.sum(np.abs(reference - candidate)))


def _save_figure(fig, save_path=None, show_plot=False, dpi=300):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def plot_joint_fusion_maps(result, title=None, save_path=None, show_plot=False, dpi=300):
    """Plot macro risk, micro safety, and their fused maps."""
    maps = [
        ("Risk probability", result["risk_probability"], "magma"),
        ("Safety coefficient", result["safety_coefficient"], "viridis"),
        ("Joint hazard", result["joint_hazard"], "inferno"),
        ("Joint safe score", result["joint_safe_score"], "viridis"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (map_title, map_data, cmap) in zip(axes.flat, maps):
        image = ax.imshow(map_data, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(map_title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _save_figure(fig, save_path=save_path, show_plot=show_plot, dpi=dpi)


def plot_joint_safe_window_map(result, title=None, save_path=None, show_plot=False, dpi=300):
    """Plot the fused joint safe-score map with the best window overlay."""
    joint_safe_score = result["joint_safe_score"]
    x_grid = result["x_grid"]
    y_grid = result["y_grid"]
    best_window = result.get("best_window")

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(
        joint_safe_score,
        origin="lower",
        extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    fig.colorbar(image, ax=ax, label="Joint safe score")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    if title:
        ax.set_title(title)

    if best_window is not None:
        dx = x_grid[0, 1] - x_grid[0, 0] if x_grid.shape[1] > 1 else 1.0
        dy = y_grid[1, 0] - y_grid[0, 0] if y_grid.shape[0] > 1 else 1.0
        rect = Rectangle(
            (x_grid[best_window["top_left_row"], best_window["top_left_col"]] - dx / 2.0,
             y_grid[best_window["top_left_row"], best_window["top_left_col"]] - dy / 2.0),
            best_window["window_cols"] * dx,
            best_window["window_rows"] * dy,
            fill=False,
            edgecolor="white",
            linewidth=2.0,
        )
        ax.add_patch(rect)
        ax.scatter(
            best_window["center_x"],
            best_window["center_y"],
            color="red",
            s=40,
            label="Best joint window",
        )
        ax.legend(loc="upper right")

    fig.tight_layout()
    _save_figure(fig, save_path=save_path, show_plot=show_plot, dpi=dpi)
