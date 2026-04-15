import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata


def infer_grid_shape(num_points, x_span, y_span):
    """Infer a raster-like grid shape from point count and XY extent."""
    if num_points < 4:
        raise ValueError("Need at least 4 points to infer a grid shape")

    target_ratio = x_span / y_span if y_span > 1e-12 else 1.0
    factor_pairs = []
    max_factor = int(np.sqrt(num_points)) + 1
    for rows in range(2, max_factor + 1):
        if num_points % rows != 0:
            continue
        cols = num_points // rows
        factor_pairs.append((rows, cols))
        factor_pairs.append((cols, rows))

    if not factor_pairs:
        side = int(np.sqrt(num_points))
        return side, max(2, num_points // max(side, 1))

    best_shape = None
    best_score = np.inf
    for rows, cols in factor_pairs:
        grid_ratio = (cols - 1) / max(rows - 1, 1)
        score = abs(grid_ratio - target_ratio)
        if score < best_score:
            best_score = score
            best_shape = (rows, cols)
    return best_shape


def build_elevation_grid(points, grid_shape=None):
    """Interpolate irregular points to a regular elevation grid."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")

    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    x_span = max(x_max - x_min, 1e-12)
    y_span = max(y_max - y_min, 1e-12)

    if grid_shape is None:
        grid_shape = infer_grid_shape(len(points), x_span, y_span)

    rows, cols = grid_shape
    x_coords = np.linspace(x_min, x_max, cols)
    y_coords = np.linspace(y_min, y_max, rows)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    z_grid = griddata(points[:, :2], points[:, 2], (x_grid, y_grid), method="linear")
    if np.isnan(z_grid).any():
        z_nearest = griddata(points[:, :2], points[:, 2], (x_grid, y_grid), method="nearest")
        z_grid = np.where(np.isnan(z_grid), z_nearest, z_grid)

    dx = x_coords[1] - x_coords[0] if cols > 1 else 1.0
    dy = y_coords[1] - y_coords[0] if rows > 1 else 1.0

    return {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "z_grid": z_grid,
        "dx": dx,
        "dy": dy,
        "grid_shape": (rows, cols),
    }


def plane_to_surface(n, d, x_grid, y_grid):
    """Evaluate plane z values on a regular grid."""
    n = np.asarray(n, dtype=float)
    if n.shape != (3,):
        raise ValueError(f"Expected normal vector with shape (3,), got {n.shape}")
    if abs(n[2]) < 1e-10:
        raise ValueError("Plane normal z component is too small to solve for z")
    return (-n[0] * x_grid - n[1] * y_grid - d) / n[2]


def plane_to_slope_intercept(n, d):
    """Convert ax + by + cz + d = 0 to z = a*x + b*y + c."""
    n = np.asarray(n, dtype=float)
    if abs(n[2]) < 1e-10:
        raise ValueError("Plane normal z component is too small to convert to z=ax+by+c form")
    a = -n[0] / n[2]
    b = -n[1] / n[2]
    c = -d / n[2]
    return a, b, c


def compute_vertical_roughness(z_grid, plane_grid):
    """Paper-style roughness: vertical distance to the fitted datum plane."""
    return np.abs(z_grid - plane_grid)


def compute_local_slope(z_grid, n, d, dx, dy):
    """Compute local slope as angle between terrain normal and plane normal."""
    dz_dy, dz_dx = np.gradient(z_grid, dy, dx)
    terrain_normals = np.stack((-dz_dx, -dz_dy, np.ones_like(z_grid)), axis=-1)

    a, b, _ = plane_to_slope_intercept(n, d)
    plane_normal = np.array([-a, -b, 1.0], dtype=float)
    plane_normal /= np.linalg.norm(plane_normal)

    terrain_norms = np.linalg.norm(terrain_normals, axis=-1)
    dots = np.abs(np.einsum("ijk,k->ij", terrain_normals, plane_normal))
    cos_theta = dots / np.maximum(terrain_norms, 1e-12)
    cos_theta = np.clip(cos_theta, 0.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def compute_safety_coefficient(roughness, slope, h_threshold=0.3, theta_threshold=10.0):
    """Compute the safety coefficient matrix C from paper Eq. (3.115)."""
    roughness = np.asarray(roughness, dtype=float)
    slope = np.asarray(slope, dtype=float)

    safe_mask = (roughness < h_threshold) & (slope < theta_threshold)
    coefficient = np.zeros_like(roughness, dtype=float)
    coefficient[safe_mask] = (
        (h_threshold - roughness[safe_mask]) * (theta_threshold - slope[safe_mask])
        / (h_threshold * theta_threshold)
    )
    return coefficient


def search_safe_windows(coefficient, x_grid, y_grid, window_size=5):
    """Search safe landing windows using the paper's min/mean strategy."""
    coefficient = np.asarray(coefficient, dtype=float)
    rows, cols = coefficient.shape

    if isinstance(window_size, int):
        window_rows = window_cols = window_size
    else:
        window_rows, window_cols = window_size

    if rows < window_rows or cols < window_cols:
        raise ValueError(
            f"Window size {(window_rows, window_cols)} is larger than coefficient grid {coefficient.shape}"
        )

    score_rows = rows - window_rows + 1
    score_cols = cols - window_cols + 1
    score_map = np.zeros((score_rows, score_cols), dtype=float)

    best_score = -1.0
    best_window = None

    for row in range(score_rows):
        for col in range(score_cols):
            window = coefficient[row : row + window_rows, col : col + window_cols]
            if np.min(window) <= 0.0:
                score = 0.0
            else:
                score = float(np.mean(window))

            score_map[row, col] = score
            if score > best_score:
                center_row = row + window_rows // 2
                center_col = col + window_cols // 2
                best_score = score
                best_window = {
                    "top_left_row": row,
                    "top_left_col": col,
                    "center_row": center_row,
                    "center_col": center_col,
                    "center_x": float(x_grid[center_row, center_col]),
                    "center_y": float(y_grid[center_row, center_col]),
                    "mean_score": score,
                    "min_score": float(np.min(window)),
                    "window_rows": window_rows,
                    "window_cols": window_cols,
                }

    return {
        "score_map": score_map,
        "best_window": best_window,
    }


def evaluate_plane_safety(
    points,
    n,
    d,
    h_threshold=0.3,
    theta_threshold=10.0,
    window_size=5,
    grid_shape=None,
):
    """Run the full paper-style safety evaluation pipeline for one fitted plane."""
    grid = build_elevation_grid(points, grid_shape=grid_shape)
    plane_grid = plane_to_surface(n, d, grid["x_grid"], grid["y_grid"])
    roughness = compute_vertical_roughness(grid["z_grid"], plane_grid)
    slope = compute_local_slope(grid["z_grid"], n, d, grid["dx"], grid["dy"])
    coefficient = compute_safety_coefficient(
        roughness,
        slope,
        h_threshold=h_threshold,
        theta_threshold=theta_threshold,
    )
    window_result = search_safe_windows(
        coefficient,
        grid["x_grid"],
        grid["y_grid"],
        window_size=window_size,
    )

    return {
        **grid,
        "plane_grid": plane_grid,
        "roughness": roughness,
        "slope": slope,
        "safety_coefficient": coefficient,
        "score_map": window_result["score_map"],
        "best_window": window_result["best_window"],
        "h_threshold": h_threshold,
        "theta_threshold": theta_threshold,
        "window_size": window_size,
        "mean_safety": float(np.mean(coefficient)),
        "max_safety": float(np.max(coefficient)),
        "min_safety": float(np.min(coefficient)),
    }


def compare_safety_maps(reference_map, candidate_map):
    """Return the sum of absolute per-pixel differences between two safety maps."""
    reference_map = np.asarray(reference_map, dtype=float)
    candidate_map = np.asarray(candidate_map, dtype=float)
    if reference_map.shape != candidate_map.shape:
        raise ValueError(
            f"Safety maps must have the same shape, got {reference_map.shape} and {candidate_map.shape}"
        )
    return float(np.sum(np.abs(reference_map - candidate_map)))


def _save_figure(fig, save_path=None, show_plot=False, dpi=300):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def plot_safety_coefficient_map(result, title=None, save_path=None, show_plot=False, dpi=300):
    """Plot the paper-style safety coefficient map with the best safe window overlay."""
    coefficient = result["safety_coefficient"]
    x_grid = result["x_grid"]
    y_grid = result["y_grid"]
    best_window = result.get("best_window")

    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = ax.imshow(
        coefficient,
        origin="lower",
        extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    plt.colorbar(heatmap, ax=ax, label="Safety Coefficient")

    if best_window and best_window["mean_score"] > 0:
        row = best_window["top_left_row"]
        col = best_window["top_left_col"]
        window_rows = best_window["window_rows"]
        window_cols = best_window["window_cols"]

        x0 = x_grid[row, col]
        y0 = y_grid[row, col]
        x1 = x_grid[min(row + window_rows - 1, x_grid.shape[0] - 1), min(col + window_cols - 1, x_grid.shape[1] - 1)]
        y1 = y_grid[min(row + window_rows - 1, y_grid.shape[0] - 1), min(col + window_cols - 1, y_grid.shape[1] - 1)]

        rect = Rectangle(
            (min(x0, x1), min(y0, y1)),
            abs(x1 - x0),
            abs(y1 - y0),
            fill=False,
            edgecolor="red",
            linewidth=2.0,
        )
        ax.add_patch(rect)
        ax.scatter(
            best_window["center_x"],
            best_window["center_y"],
            c="white",
            s=50,
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )

    ax.set_title(title or "Safety Coefficient Map")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    _save_figure(fig, save_path=save_path, show_plot=show_plot, dpi=dpi)


def plot_window_score_map(result, title=None, save_path=None, show_plot=False, dpi=300):
    """Plot the safe-window score map."""
    score_map = result["score_map"]
    x_grid = result["x_grid"]
    y_grid = result["y_grid"]
    best_window = result.get("best_window")

    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = ax.imshow(
        score_map,
        origin="lower",
        extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
        cmap="magma",
        aspect="auto",
    )
    plt.colorbar(heatmap, ax=ax, label="Window Safety Score")

    if best_window and best_window["mean_score"] > 0:
        ax.scatter(
            best_window["center_x"],
            best_window["center_y"],
            c="cyan",
            s=60,
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )

    ax.set_title(title or "Safe Window Score Map")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    _save_figure(fig, save_path=save_path, show_plot=show_plot, dpi=dpi)
