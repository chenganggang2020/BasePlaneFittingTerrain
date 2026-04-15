import numpy as np
from scipy.ndimage import convolve, maximum_filter, minimum_filter, uniform_filter

from safety_evaluation import build_elevation_grid


FEATURE_NAMES = [
    "weighted_local_variance",
    "absolute_tpi",
    "local_mean_slope",
    "elevation_range",
]


def _ensure_odd_window(window_size):
    window_size = int(max(3, window_size))
    return window_size if window_size % 2 == 1 else window_size + 1


def _gaussian_kernel(window_size, sigma):
    window_size = _ensure_odd_window(window_size)
    radius = window_size // 2
    sigma = float(sigma) if sigma is not None else max(window_size / 3.0, 1.0)
    coords = np.arange(-radius, radius + 1, dtype=float)
    xx, yy = np.meshgrid(coords, coords)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel_sum = np.sum(kernel)
    if kernel_sum <= 0:
        return np.full((window_size, window_size), 1.0 / (window_size ** 2), dtype=float)
    return kernel / kernel_sum


def normalize_feature_map(feature_map, lower_quantile=0.02, upper_quantile=0.98):
    """Robustly normalize a feature map to [0, 1]."""
    feature_map = np.asarray(feature_map, dtype=float)
    valid_mask = np.isfinite(feature_map)
    if not np.any(valid_mask):
        return np.zeros_like(feature_map, dtype=float)

    valid_values = feature_map[valid_mask]
    low = np.quantile(valid_values, lower_quantile)
    high = np.quantile(valid_values, upper_quantile)
    if high - low < 1e-12:
        normalized = np.zeros_like(feature_map, dtype=float)
    else:
        normalized = (feature_map - low) / (high - low)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized[~valid_mask] = 0.0
    return normalized


def compute_weighted_local_variance(z_grid, window_size=5, sigma=None):
    """Compute Gaussian-weighted local elevation variance."""
    kernel = _gaussian_kernel(window_size, sigma)
    local_mean = convolve(z_grid, kernel, mode="nearest")
    local_mean_sq = convolve(z_grid ** 2, kernel, mode="nearest")
    variance = np.maximum(local_mean_sq - local_mean ** 2, 0.0)
    return variance


def compute_tpi(z_grid, window_size=5):
    """Compute topographic position index."""
    window_size = _ensure_odd_window(window_size)
    local_mean = uniform_filter(z_grid, size=window_size, mode="nearest")
    return z_grid - local_mean


def compute_slope_magnitude(z_grid, dx, dy):
    """Compute local slope magnitude in degrees."""
    dz_dy, dz_dx = np.gradient(z_grid, dy, dx)
    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    return np.degrees(slope_rad)


def compute_local_mean_slope(z_grid, dx, dy, window_size=5):
    """Compute the mean slope within a local window."""
    window_size = _ensure_odd_window(window_size)
    slope = compute_slope_magnitude(z_grid, dx, dy)
    return uniform_filter(slope, size=window_size, mode="nearest")


def compute_elevation_range(z_grid, window_size=5):
    """Compute the local elevation range."""
    window_size = _ensure_odd_window(window_size)
    local_max = maximum_filter(z_grid, size=window_size, mode="nearest")
    local_min = minimum_filter(z_grid, size=window_size, mode="nearest")
    return local_max - local_min


def extract_terrain_feature_stack(points, grid_shape=None, window_size=5, variance_sigma=None):
    """Extract the thesis-style terrain feature stack from point clouds."""
    grid = build_elevation_grid(points, grid_shape=grid_shape)
    z_grid = grid["z_grid"]
    dx = grid["dx"]
    dy = grid["dy"]

    weighted_variance = compute_weighted_local_variance(
        z_grid,
        window_size=window_size,
        sigma=variance_sigma,
    )
    tpi = compute_tpi(z_grid, window_size=window_size)
    absolute_tpi = np.abs(tpi)
    local_mean_slope = compute_local_mean_slope(z_grid, dx, dy, window_size=window_size)
    elevation_range = compute_elevation_range(z_grid, window_size=window_size)

    features_raw = {
        "weighted_local_variance": weighted_variance,
        "absolute_tpi": absolute_tpi,
        "local_mean_slope": local_mean_slope,
        "elevation_range": elevation_range,
    }
    features_normalized = {
        name: normalize_feature_map(feature_map)
        for name, feature_map in features_raw.items()
    }
    feature_stack = np.stack([features_normalized[name] for name in FEATURE_NAMES], axis=-1)

    return {
        **grid,
        "features_raw": features_raw,
        "features_normalized": features_normalized,
        "feature_stack": feature_stack,
        "feature_names": FEATURE_NAMES.copy(),
        "window_size": _ensure_odd_window(window_size),
        "variance_sigma": variance_sigma,
    }
