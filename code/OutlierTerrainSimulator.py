import argparse
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
from scipy.ndimage import gaussian_filter
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

try:
    from osgeo import gdal, osr
except ImportError:  # pragma: no cover - optional dependency
    gdal = None
    osr = None


class OutlierTerrainSimulator:
    """
    Generate synthetic sloped DEMs and labeled point clouds with both geometric
    obstacles and structured sensing artefacts.

    Compared with the earlier simulator, this version adds:
    1. Stripe artefacts that mimic scan-line corruption.
    2. Contiguous bad batches that mimic consecutive faulty scans.
    3. Range-dependent measurement noise on the exported point cloud.
    """

    def __init__(
        self,
        size=10,
        pixel_size=0.3,
        slope_angle=0.0,
        threshold=0.1,
        z=0.0,
        z_noise_std=0.05,
        xy_noise_std=0.03,
        crater_outlier_share=0.45,
        rock_outlier_share=0.15,
        stripe_outlier_share=0.20,
        bad_batch_outlier_share=0.15,
        random_outlier_share=0.05,
        stripe_axis="y",
        stripe_width_range=(1, 4),
        stripe_bias_range=(0.10, 0.30),
        stripe_jitter_std=0.03,
        bad_batch_axis="y",
        bad_batch_length_range=(3, 8),
        bad_batch_bias_range=(0.12, 0.35),
        bad_batch_jitter_std=0.04,
        range_noise_gain=1.50,
        range_xy_gain=0.75,
        range_noise_power=2.0,
        random_state=42,
    ):
        self.size = float(size)
        self.half_size = self.size / 2.0
        self.pixel_size = float(pixel_size)
        self.slope_angle = float(slope_angle)
        self.threshold = float(threshold)
        self.base_z = float(z)
        self.z_noise_std = float(z_noise_std)
        self.xy_noise_std = float(xy_noise_std)

        self.stripe_axis = stripe_axis.lower()
        self.stripe_width_range = tuple(int(v) for v in stripe_width_range)
        self.stripe_bias_range = tuple(float(v) for v in stripe_bias_range)
        self.stripe_jitter_std = float(stripe_jitter_std)

        self.bad_batch_axis = bad_batch_axis.lower()
        self.bad_batch_length_range = tuple(int(v) for v in bad_batch_length_range)
        self.bad_batch_bias_range = tuple(float(v) for v in bad_batch_bias_range)
        self.bad_batch_jitter_std = float(bad_batch_jitter_std)

        self.range_noise_gain = float(range_noise_gain)
        self.range_xy_gain = float(range_xy_gain)
        self.range_noise_power = float(range_noise_power)
        self.random_state = int(random_state)
        self.rng = np.random.default_rng(self.random_state)

        self.outlier_shares = self._normalize_share_profile(
            crater=crater_outlier_share,
            rock=rock_outlier_share,
            stripe=stripe_outlier_share,
            bad_batch=bad_batch_outlier_share,
            random=random_outlier_share,
        )

        self.base_dem, self.xx, self.yy = self._generate_base_terrain()
        self.range_ratio_map = self._build_range_ratio_map()
        self.rows, self.cols = self.base_dem.shape
        self.total_points = self.rows * self.cols
        self.actual_dem = self.base_dem.copy()

    @staticmethod
    def _normalize_share_profile(**shares):
        total = sum(max(float(value), 0.0) for value in shares.values())
        if total <= 0:
            uniform = 1.0 / max(len(shares), 1)
            return {name: uniform for name in shares}
        return {name: max(float(value), 0.0) / total for name, value in shares.items()}

    def _generate_base_terrain(self):
        x = np.arange(-self.half_size, self.half_size + self.pixel_size, self.pixel_size)
        y = np.arange(-self.half_size, self.half_size + self.pixel_size, self.pixel_size)
        xx, yy = np.meshgrid(x, y)

        slope_rad = np.radians(self.slope_angle)
        dem = np.tan(slope_rad) * xx + self.base_z

        if self.z_noise_std > 0:
            smooth_noise = gaussian_filter(
                self.rng.normal(0.0, self.z_noise_std, size=xx.shape),
                sigma=0.8,
            )
            dem = dem + smooth_noise

        return dem, xx, yy

    def _build_range_ratio_map(self):
        radial_distance = np.sqrt(self.xx ** 2 + self.yy ** 2)
        return radial_distance / max(np.max(radial_distance), 1e-12)

    def _component_budget(self, target_outliers, component_name):
        return int(round(target_outliers * self.outlier_shares[component_name]))

    def _random_signed_amplitude(self, amplitude_range):
        amplitude = self.rng.uniform(*amplitude_range)
        return amplitude * self.rng.choice([-1.0, 1.0])

    def _generate_crater(self, dem, x0, y0, diameter, impact_angle=60):
        radius = diameter / 2.0
        depth = min(1.5, diameter * 0.6)
        rim_height = depth * 0.4

        theta = np.radians(impact_angle)
        x_rot = (self.xx - x0) * np.cos(theta) + (self.yy - y0) * np.sin(theta)
        y_rot = -(self.xx - x0) * np.sin(theta) + (self.yy - y0) * np.cos(theta)

        mask_scale = 1.1 - (diameter - 0.3) * 0.08 / (5.0 - 0.3)
        crater_mask = (x_rot ** 2 + y_rot ** 2) <= (radius * mask_scale) ** 2

        radial = np.sqrt(x_rot ** 2 + y_rot ** 2) / max(radius, 1e-12)
        radial = np.where(crater_mask, radial, 0.0)

        dem[crater_mask] -= depth * (1.0 - radial[crater_mask] ** 2)

        rim_mask = crater_mask & (radial >= 0.6)
        dem[rim_mask] += rim_height * (1.0 - (radial[rim_mask] - 0.6) / 0.5) ** 2
        dem[:] = gaussian_filter(dem, sigma=0.2)
        return crater_mask

    def _generate_rock(self, dem, x0, y0, radius, min_height=0.1):
        height = max(radius * self.rng.uniform(0.5, 0.8), min_height)
        radial = np.sqrt((self.xx - x0) ** 2 + (self.yy - y0) ** 2)
        rock_mask = radial <= radius * 0.9

        dem[rock_mask] += height * np.exp(-(radial[rock_mask] / max(radius, 1e-12)) ** 3)
        dem[:] = gaussian_filter(dem, sigma=0.15)
        return rock_mask

    def _generate_stripe_artifact(self, dem):
        axis = self.stripe_axis if self.stripe_axis in {"x", "y"} else "y"
        width_min, width_max = self.stripe_width_range
        width = int(self.rng.integers(width_min, width_max + 1))
        signed_bias = self._random_signed_amplitude(self.stripe_bias_range)

        if axis == "x":
            start = int(self.rng.integers(0, max(self.cols - width + 1, 1)))
            mask = np.zeros_like(dem, dtype=bool)
            mask[:, start : start + width] = True
        else:
            start = int(self.rng.integers(0, max(self.rows - width + 1, 1)))
            mask = np.zeros_like(dem, dtype=bool)
            mask[start : start + width, :] = True

        stripe_texture = gaussian_filter(
            self.rng.normal(0.0, self.stripe_jitter_std, size=dem.shape),
            sigma=0.6,
        )
        dem[mask] += (signed_bias + stripe_texture[mask]) * (1.0 + 0.35 * self.range_ratio_map[mask])
        return mask

    def _generate_bad_batch(self, dem):
        axis = self.bad_batch_axis if self.bad_batch_axis in {"x", "y"} else "y"
        len_min, len_max = self.bad_batch_length_range
        batch_len = int(self.rng.integers(len_min, len_max + 1))
        signed_bias = self._random_signed_amplitude(self.bad_batch_bias_range)

        if axis == "x":
            start = int(self.rng.integers(0, max(self.cols - batch_len + 1, 1)))
            cols = np.arange(start, min(start + batch_len, self.cols))
            phase = self.rng.uniform(0.0, 2.0 * np.pi)
            waveform = np.sin(np.linspace(0.0, 2.0 * np.pi, self.rows) + phase)
            for offset, col in enumerate(cols):
                local_bias = signed_bias + self.rng.normal(0.0, self.bad_batch_jitter_std)
                dem[:, col] += local_bias + 0.35 * self.bad_batch_jitter_std * waveform
            mask = np.zeros_like(dem, dtype=bool)
            mask[:, cols] = True
        else:
            start = int(self.rng.integers(0, max(self.rows - batch_len + 1, 1)))
            rows = np.arange(start, min(start + batch_len, self.rows))
            phase = self.rng.uniform(0.0, 2.0 * np.pi)
            waveform = np.sin(np.linspace(0.0, 2.0 * np.pi, self.cols) + phase)
            for offset, row in enumerate(rows):
                local_bias = signed_bias + self.rng.normal(0.0, self.bad_batch_jitter_std)
                dem[row, :] += local_bias + 0.35 * self.bad_batch_jitter_std * waveform
            mask = np.zeros_like(dem, dtype=bool)
            mask[rows, :] = True

        return mask

    def _add_random_spikes(self, labels, target_count):
        inlier_indices = np.where(labels == 1)
        available = len(inlier_indices[0])
        if available == 0 or target_count <= 0:
            return 0

        actual_count = min(int(target_count), available)
        selected_idx = self.rng.choice(available, actual_count, replace=False)
        spike = self.rng.uniform(0.2, 0.5, size=actual_count)
        row_idx = inlier_indices[0][selected_idx]
        col_idx = inlier_indices[1][selected_idx]
        self.actual_dem[row_idx, col_idx] += spike
        return actual_count

    def calculate_outliers(self, actual_dem, base_dem):
        distance = np.abs(actual_dem - base_dem)
        return np.where(distance > self.threshold, 0, 1)

    def _save_geotiff(self, data, output_path):
        if gdal is None or osr is None:
            fallback_path = os.path.splitext(output_path)[0] + ".npy"
            np.save(fallback_path, data.astype(np.float32))
            print(f"Warning: GDAL unavailable, DEM saved as NumPy array: {fallback_path}")
            return

        rows, cols = data.shape
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)

        srs = osr.SpatialReference()
        srs.SetWellKnownGeogCS("WGS84")
        out_ds.SetProjection(srs.ExportToWkt())

        origin_x = -self.half_size
        origin_y = self.half_size
        geotransform = (origin_x, self.pixel_size, 0.0, origin_y, 0.0, -self.pixel_size)
        out_ds.SetGeoTransform(geotransform)

        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(data.astype(np.float32))
        out_band.FlushCache()
        out_ds = None

    def _save_point_cloud(self, dem, labels, output_path):
        x_nominal = self.xx.flatten()
        y_nominal = self.yy.flatten()
        z_nominal = dem.flatten()
        labels_flat = labels.flatten()
        range_ratio_flat = self.range_ratio_map.flatten()

        distance_scale = 1.0 + self.range_noise_gain * (range_ratio_flat ** self.range_noise_power)
        xy_scale = 1.0 + self.range_xy_gain * (range_ratio_flat ** self.range_noise_power)

        dx = self.rng.normal(0.0, self.xy_noise_std * xy_scale)
        dy = self.rng.normal(0.0, self.xy_noise_std * xy_scale)
        dz = self.rng.normal(0.0, self.z_noise_std * distance_scale)

        point_cloud = np.column_stack(
            (
                x_nominal + dx,
                y_nominal + dy,
                z_nominal + dz,
                labels_flat,
            )
        )
        np.savetxt(output_path, point_cloud, fmt="%.4f %.4f %.4f %d")

    def _apply_component_with_budget(self, component_name, budget, current_outliers, generator, max_attempts):
        if budget <= 0:
            return current_outliers, 0

        start_outliers = current_outliers
        target_total = min(self.total_points, start_outliers + int(budget))
        attempts = 0

        while current_outliers < target_total and attempts < max_attempts:
            attempts += 1
            dem_before = self.actual_dem.copy()
            generator(self.actual_dem)
            new_labels = self.calculate_outliers(self.actual_dem, self.base_dem)
            new_outliers = int(np.sum(new_labels == 0))

            if current_outliers < new_outliers <= target_total:
                current_outliers = new_outliers
            else:
                self.actual_dem = dem_before

        added = current_outliers - start_outliers
        print(f"  > {component_name}: added {added} outliers (target {budget})")
        return current_outliers, added

    def _sample_crater(self, dem, target_outliers):
        ratio_factor = min(target_outliers / 1000.0, 0.9)
        min_dia = 0.5 + (ratio_factor - 0.1) * (2.0 - 0.5) / 0.8
        max_dia = 1.0 + (ratio_factor - 0.1) * (10.0 - 1.0) / 0.8
        diameter = self.rng.uniform(min_dia, max_dia)
        margin = diameter * 0.05
        x0 = self.rng.uniform(-self.half_size + margin, self.half_size - margin)
        y0 = self.rng.uniform(-self.half_size + margin, self.half_size - margin)
        impact_angle = self.rng.uniform(40.0, 80.0)
        self._generate_crater(dem, x0, y0, diameter, impact_angle=impact_angle)

    def _sample_rock(self, dem, target_outliers):
        ratio_factor = min(target_outliers / 1000.0, 0.9)
        min_rad = 0.1 + (ratio_factor - 0.1) * (0.3 - 0.1) / 0.8
        max_rad = 0.5 + (ratio_factor - 0.1) * (0.8 - 0.5) / 0.8
        radius = self.rng.uniform(min_rad, max_rad)
        margin = radius * 0.3
        x0 = self.rng.uniform(-self.half_size + margin, self.half_size - margin)
        y0 = self.rng.uniform(-self.half_size + margin, self.half_size - margin)
        self._generate_rock(dem, x0, y0, radius)

    def generate_simulation(self, outlier_ratio, dem_dir, pc_dir):
        self.actual_dem = self.base_dem.copy()
        os.makedirs(dem_dir, exist_ok=True)
        os.makedirs(pc_dir, exist_ok=True)

        target_outliers = int(round(self.total_points * float(outlier_ratio)))
        budgets = {
            name: self._component_budget(target_outliers, name)
            for name in self.outlier_shares
        }

        assigned = sum(budgets.values())
        if assigned != target_outliers:
            budgets["random"] += target_outliers - assigned

        print(f"\n--- Target outlier ratio: {outlier_ratio:.2f} ({target_outliers} points) ---")
        print(f"  > Budget split: {budgets}")

        current_outliers = 0
        current_outliers, crater_added = self._apply_component_with_budget(
            "craters",
            budgets["crater"],
            current_outliers,
            lambda dem: self._sample_crater(dem, target_outliers),
            max_attempts=5000,
        )
        current_outliers, rock_added = self._apply_component_with_budget(
            "rocks",
            budgets["rock"],
            current_outliers,
            lambda dem: self._sample_rock(dem, target_outliers),
            max_attempts=4000,
        )
        current_outliers, stripe_added = self._apply_component_with_budget(
            "scan stripes",
            budgets["stripe"],
            current_outliers,
            self._generate_stripe_artifact,
            max_attempts=3000,
        )
        current_outliers, batch_added = self._apply_component_with_budget(
            "bad batches",
            budgets["bad_batch"],
            current_outliers,
            self._generate_bad_batch,
            max_attempts=3000,
        )

        remaining_needed = max(target_outliers - current_outliers, 0)
        if remaining_needed > 0:
            labels_before_random = self.calculate_outliers(self.actual_dem, self.base_dem)
            self._add_random_spikes(labels_before_random, remaining_needed)

        final_labels = self.calculate_outliers(self.actual_dem, self.base_dem)
        current_outliers = int(np.sum(final_labels == 0))
        random_added = max(current_outliers - crater_added - rock_added - stripe_added - batch_added, 0)

        pct = int(round(float(outlier_ratio) * 100))
        dem_path = os.path.join(dem_dir, f"dem_{pct}pct.tif")
        pc_path = os.path.join(pc_dir, f"point_cloud_{pct}pct.txt")

        self._save_geotiff(self.actual_dem, dem_path)
        self._save_point_cloud(self.actual_dem, final_labels, pc_path)

        actual_ratio = 100.0 * current_outliers / max(self.total_points, 1)
        print(
            "  Final composition: "
            f"crater={crater_added}, rock={rock_added}, stripe={stripe_added}, "
            f"bad_batch={batch_added}, random={random_added}"
        )
        print(f"  Final result: {current_outliers} outliers ({actual_ratio:.1f}%)")


def build_default_structured_simulator_kwargs(slope_angle=35.0, base_z=30.0):
    return {
        "size": 10,
        "slope_angle": slope_angle,
        "z": base_z,
        "threshold": 0.05,
        "z_noise_std": 0.08,
        "xy_noise_std": 0.05,
        "stripe_outlier_share": 0.18,
        "bad_batch_outlier_share": 0.17,
        "range_noise_gain": 1.8,
        "range_xy_gain": 0.9,
    }


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


def _simulate_ratio_worker(simulator_kwargs, outlier_ratio, dem_dir, pc_dir):
    simulator = OutlierTerrainSimulator(**simulator_kwargs)
    simulator.generate_simulation(outlier_ratio, dem_dir, pc_dir)
    return outlier_ratio


def generate_structured_dataset(
    dem_dir,
    pc_dir,
    ratio_start=0.10,
    ratio_stop=0.91,
    ratio_step=0.01,
    simulator_kwargs=None,
    max_workers=1,
):
    simulator_kwargs = dict(simulator_kwargs or {})
    os.makedirs(dem_dir, exist_ok=True)
    os.makedirs(pc_dir, exist_ok=True)

    ratio_values = [float(round(value, 6)) for value in np.arange(ratio_start, ratio_stop, ratio_step)]
    if not ratio_values:
        print("No ratios scheduled for structured dataset generation.")
        return []

    max_workers = min(_resolve_max_workers(max_workers), len(ratio_values))
    base_seed = int(simulator_kwargs.get("random_state", 42))

    if max_workers <= 1:
        for idx, ratio in enumerate(tqdm(ratio_values, desc="Generating structured data")):
            worker_kwargs = dict(simulator_kwargs)
            worker_kwargs["random_state"] = base_seed + idx * 997 + int(round(ratio * 10000))
            _simulate_ratio_worker(worker_kwargs, ratio, dem_dir, pc_dir)
    else:
        executor, backend = _build_parallel_executor(max_workers)
        print(f"Generating structured dataset with {max_workers} {backend} workers.")
        with executor:
            future_to_ratio = {}
            for idx, ratio in enumerate(ratio_values):
                worker_kwargs = dict(simulator_kwargs)
                worker_kwargs["random_state"] = base_seed + idx * 997 + int(round(ratio * 10000))
                future = executor.submit(_simulate_ratio_worker, worker_kwargs, ratio, dem_dir, pc_dir)
                future_to_ratio[future] = ratio

            progress = tqdm(total=len(ratio_values), desc="Generating structured data")
            try:
                for future in as_completed(future_to_ratio):
                    ratio = future_to_ratio[future]
                    future.result()
                    progress.update(1)
            finally:
                progress.close()

    return ratio_values


def parse_args():
    parser = argparse.ArgumentParser(description="Generate structured synthetic terrain and point clouds.")
    parser.add_argument("--dem-dir", default="data/dem_files_output_structured35")
    parser.add_argument("--pc-dir", default="data/point_clouds_output_structured35")
    parser.add_argument("--slope-angle", type=float, default=35.0)
    parser.add_argument("--base-z", type=float, default=30.0)
    parser.add_argument("--ratio-start", type=float, default=0.10)
    parser.add_argument("--ratio-stop", type=float, default=0.91)
    parser.add_argument("--ratio-step", type=float, default=0.01)
    parser.add_argument(
        "--workers",
        default=1,
        help="Number of worker processes for ratio-level parallel generation. Use 0 or 'auto' for CPU-count-based auto mode.",
    )
    return parser.parse_args()


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    args = parse_args()
    simulator_kwargs = build_default_structured_simulator_kwargs(
        slope_angle=args.slope_angle,
        base_z=args.base_z,
    )
    generate_structured_dataset(
        dem_dir=args.dem_dir,
        pc_dir=args.pc_dir,
        ratio_start=args.ratio_start,
        ratio_stop=args.ratio_stop,
        ratio_step=args.ratio_step,
        simulator_kwargs=simulator_kwargs,
        max_workers=args.workers,
    )

    print(
        "\nStructured synthetic terrain generation finished:\n"
        f"DEM directory: {os.path.abspath(args.dem_dir)}\n"
        f"Point-cloud directory: {os.path.abspath(args.pc_dir)}"
    )


if __name__ == "__main__":
    main()
