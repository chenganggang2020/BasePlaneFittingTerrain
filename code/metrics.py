import numpy as np


def normalize_inlier_mask(predicted_inliers, point_count):
    """Normalize inlier outputs to a boolean mask of length ``point_count``."""
    point_count = int(point_count)
    if point_count <= 0:
        return np.array([], dtype=bool)
    if predicted_inliers is None:
        return None

    arr = np.asarray(predicted_inliers)
    if arr.size == 0:
        return np.zeros(point_count, dtype=bool)

    if np.issubdtype(arr.dtype, np.bool_):
        if arr.size != point_count:
            return None
        return arr.reshape(-1).astype(bool, copy=False)

    if np.issubdtype(arr.dtype, np.integer):
        flat = arr.reshape(-1)
        if flat.size == point_count and np.all(np.isin(flat, [0, 1])):
            return flat.astype(bool)
        if not np.all((flat >= 0) & (flat < point_count)):
            return None
        mask = np.zeros(point_count, dtype=bool)
        mask[flat] = True
        return mask

    return None


def calculate_detection_metrics(true_labels, predicted_inliers):
    """Compute accuracy, false-negative rate, and false-positive rate."""
    true_labels = np.asarray(true_labels, dtype=bool)
    predicted_mask = normalize_inlier_mask(predicted_inliers, len(true_labels))
    if predicted_mask is None or predicted_mask.size == 0:
        return {
            "Accuracy": np.nan,
            "False Negative Rate": np.nan,
            "False Positive Rate": np.nan,
        }

    tp = np.sum(true_labels & predicted_mask)
    tn = np.sum(~true_labels & ~predicted_mask)
    fp = np.sum(~true_labels & predicted_mask)
    fn = np.sum(true_labels & ~predicted_mask)

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    return {
        "Accuracy": accuracy,
        "False Negative Rate": fnr,
        "False Positive Rate": fpr,
    }


def calculate_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return the acute angle in degrees between two vectors."""
    if v1 is None or v2 is None:
        return np.nan
    try:
        v1 = np.asarray(v1, dtype=float)
        v2 = np.asarray(v2, dtype=float)
    except (TypeError, ValueError):
        return np.nan

    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm <= 0 or v2_norm <= 0:
        return np.nan

    dot_product = np.dot(v1, v2)
    cos_theta = np.abs(dot_product) / (v1_norm * v2_norm)
    angle_rad = np.arccos(np.clip(cos_theta, 0.0, 1.0))
    return float(np.degrees(angle_rad))


def calculate_d_est_difference(d_est, initial_d_est=0.0):
    """Return the absolute difference between the fitted and reference intercepts."""
    if d_est is None or initial_d_est is None:
        return np.nan
    try:
        d_est = float(d_est)
        initial_d_est = float(initial_d_est)
    except (TypeError, ValueError):
        return np.nan
    return abs(d_est - initial_d_est)
