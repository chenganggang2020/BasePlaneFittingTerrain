import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from terrain_features import FEATURE_NAMES, extract_terrain_feature_stack


def _sigmoid(values):
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-values))


def _logit(probabilities, eps=1e-6):
    probabilities = np.clip(probabilities, eps, 1.0 - eps)
    return np.log(probabilities / (1.0 - probabilities))


def build_label_grid(points, labels, x_grid, y_grid):
    """Map point-wise labels to the analysis grid by nearest-neighbor interpolation."""
    if labels is None:
        return None
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.shape[0] != points.shape[0]:
        return None
    label_grid = griddata(points[:, :2], labels.astype(float), (x_grid, y_grid), method="nearest")
    if label_grid is None:
        return None
    return np.rint(label_grid).astype(int)


def compute_multi_threshold_map(feature_stack, thresholds=(0.35, 0.5, 0.65)):
    """Vote-based risk map that mimics multi-threshold joint decisions."""
    votes = []
    for threshold in thresholds:
        votes.append((feature_stack >= threshold).astype(float))
    stacked_votes = np.stack(votes, axis=0)
    return np.mean(stacked_votes, axis=(0, 3))


def compute_feature_probabilities(feature_stack, midpoint=0.5, steepness=8.0):
    """Convert normalized feature magnitudes to hazard probabilities."""
    return _sigmoid((feature_stack - midpoint) * steepness)


def compute_covariance_corrected_weights(feature_stack):
    """Down-weight strongly correlated terrain features."""
    n_features = feature_stack.shape[-1]
    flattened = feature_stack.reshape(-1, n_features)
    if flattened.shape[0] < 2:
        return np.full(n_features, 1.0 / n_features), np.eye(n_features)

    corr = np.corrcoef(flattened, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    mean_abs_corr = np.mean(np.abs(corr - np.eye(n_features)), axis=1)
    raw_weights = 1.0 / (1.0 + mean_abs_corr)
    weight_sum = np.sum(raw_weights)
    if weight_sum <= 0:
        return np.full(n_features, 1.0 / n_features), corr
    return raw_weights / weight_sum, corr


def combine_feature_probabilities(feature_probabilities, prior=0.35, weights=None):
    """Combine feature hazard probabilities with a weighted logit fusion rule."""
    n_features = feature_probabilities.shape[-1]
    if weights is None:
        weights = np.full(n_features, 1.0 / n_features, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)

    fused_logit = _logit(prior) + np.tensordot(_logit(feature_probabilities), weights, axes=([-1], [0]))
    return _sigmoid(fused_logit)


def smooth_probability_map(probability_map, smoothness=0.65, num_iters=6):
    """Approximate the paper's MRF-style spatial smoothing with neighbor-consistent refinement."""
    smoothness = float(np.clip(smoothness, 0.0, 1.0))
    probability_map = np.asarray(probability_map, dtype=float)
    current = probability_map.copy()
    for _ in range(int(max(0, num_iters))):
        neighbor_mean = (
            np.roll(current, 1, axis=0)
            + np.roll(current, -1, axis=0)
            + np.roll(current, 1, axis=1)
            + np.roll(current, -1, axis=1)
        ) / 4.0
        current = np.clip((1.0 - smoothness) * probability_map + smoothness * neighbor_mean, 0.0, 1.0)
    return current


def evaluate_risk_map_against_labels(risk_probability, label_grid, threshold=0.5):
    """Compare high-risk predictions to simulator labels when available."""
    if label_grid is None:
        return {}

    true_hazard = label_grid == 0
    predicted_hazard = risk_probability >= threshold

    tp = int(np.sum(true_hazard & predicted_hazard))
    tn = int(np.sum(~true_hazard & ~predicted_hazard))
    fp = int(np.sum(~true_hazard & predicted_hazard))
    fn = int(np.sum(true_hazard & ~predicted_hazard))
    total = tp + tn + fp + fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    accuracy = (tp + tn) / total if total > 0 else np.nan
    f1_score = (
        2.0 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else np.nan
    )
    return {
        "Risk Map Accuracy": accuracy,
        "Risk Map Precision": precision,
        "Risk Map Recall": recall,
        "Risk Map F1": f1_score,
        "Hazard Cell Ratio": float(np.mean(true_hazard)),
    }


def compute_risk_probability_map(
    points,
    labels=None,
    grid_shape=None,
    window_size=5,
    variance_sigma=None,
    prior=0.35,
    multi_thresholds=(0.35, 0.5, 0.65),
    feature_midpoint=0.5,
    feature_steepness=8.0,
    mrf_lambda=0.65,
    mrf_iters=6,
):
    """Run the thesis-style large-area risk probability pipeline."""
    feature_result = extract_terrain_feature_stack(
        points,
        grid_shape=grid_shape,
        window_size=window_size,
        variance_sigma=variance_sigma,
    )
    feature_stack = feature_result["feature_stack"]
    feature_probabilities = compute_feature_probabilities(
        feature_stack,
        midpoint=feature_midpoint,
        steepness=feature_steepness,
    )
    equal_weights = np.full(feature_stack.shape[-1], 1.0 / feature_stack.shape[-1], dtype=float)
    covariance_weights, correlation_matrix = compute_covariance_corrected_weights(feature_stack)

    multi_threshold_map = compute_multi_threshold_map(feature_stack, thresholds=multi_thresholds)
    weighted_linear_risk = np.tensordot(feature_stack, covariance_weights, axes=([-1], [0]))
    bayes_probability = combine_feature_probabilities(
        feature_probabilities,
        prior=prior,
        weights=equal_weights,
    )
    covariance_corrected_probability = combine_feature_probabilities(
        feature_probabilities,
        prior=prior,
        weights=covariance_weights,
    )
    smoothed_probability = smooth_probability_map(
        covariance_corrected_probability,
        smoothness=mrf_lambda,
        num_iters=mrf_iters,
    )
    risk_probability = np.clip(
        0.2 * multi_threshold_map
        + 0.25 * weighted_linear_risk
        + 0.55 * smoothed_probability,
        0.0,
        1.0,
    )

    label_grid = build_label_grid(
        points=np.asarray(points, dtype=float),
        labels=labels,
        x_grid=feature_result["x_grid"],
        y_grid=feature_result["y_grid"],
    )
    metrics = evaluate_risk_map_against_labels(risk_probability, label_grid)

    return {
        **feature_result,
        "feature_probabilities": {
            FEATURE_NAMES[idx]: feature_probabilities[:, :, idx]
            for idx in range(len(FEATURE_NAMES))
        },
        "multi_threshold_map": multi_threshold_map,
        "weighted_linear_risk": weighted_linear_risk,
        "bayes_probability": bayes_probability,
        "covariance_corrected_probability": covariance_corrected_probability,
        "smoothed_probability": smoothed_probability,
        "risk_probability": risk_probability,
        "label_grid": label_grid,
        "covariance_weights": covariance_weights,
        "correlation_matrix": correlation_matrix,
        "prior": prior,
        "mrf_lambda": mrf_lambda,
        "mrf_iters": mrf_iters,
        "high_risk_ratio": float(np.mean(risk_probability >= 0.5)),
        "mean_risk_probability": float(np.mean(risk_probability)),
        "max_risk_probability": float(np.max(risk_probability)),
        "metrics": metrics,
    }


def _save_figure(fig, save_path=None, show_plot=False, dpi=300):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def _style_map_axis(ax, title):
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel("X", fontsize=9)
    ax.set_ylabel("Y", fontsize=9)
    ax.tick_params(labelsize=8, length=2.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_feature_maps(result, save_path=None, show_plot=False, dpi=300):
    """Plot the four core terrain features."""
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.0))
    for ax, feature_name in zip(axes.flat, FEATURE_NAMES):
        image = ax.imshow(result["features_normalized"][feature_name], origin="lower", cmap="viridis")
        _style_map_axis(ax, feature_name.replace("_", " ").title())
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Terrain Feature Maps", fontsize=14, y=0.97, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.07, top=0.92, wspace=0.20, hspace=0.22)
    _save_figure(fig, save_path=save_path, show_plot=show_plot, dpi=dpi)


def plot_risk_fusion_maps(result, save_path=None, show_plot=False, dpi=300):
    """Plot the intermediate and final risk fusion stages."""
    maps = [
        ("Multi-threshold", result["multi_threshold_map"]),
        ("Weighted linear", result["weighted_linear_risk"]),
        ("Bayes fusion", result["bayes_probability"]),
        ("Covariance corrected", result["covariance_corrected_probability"]),
        ("MRF-style smoothing", result["smoothed_probability"]),
        ("Final risk probability", result["risk_probability"]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.4))
    for ax, (title, map_data) in zip(axes.flat, maps):
        image = ax.imshow(map_data, origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
        _style_map_axis(ax, title)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Risk Probability Fusion", fontsize=14, y=0.97, fontweight="bold")
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.07, top=0.92, wspace=0.18, hspace=0.24)
    _save_figure(fig, save_path=save_path, show_plot=show_plot, dpi=dpi)
