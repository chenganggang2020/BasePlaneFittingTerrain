import argparse
import json
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
        macro_relief_std=0.03,
        macro_relief_sigma=1.6,
        micro_relief_std=0.01,
        micro_relief_sigma=0.45,
        corrugation_amplitude=0.02,
        corrugation_wavelength=3.5,
        landing_zone_radius=1.4,
        landing_zone_suppression=0.75,
        hazard_cluster_count=3,
        hazard_cluster_spread=1.1,
        hazard_cluster_bias=0.70,
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
        sensor_height=14.0,
        sensor_offset_x=0.8,
        sensor_offset_y=-1.6,
        sensor_focus_gain=0.08,
        min_detection_probability=0.32,
        incidence_dropout_strength=0.42,
        range_dropout_strength=0.18,
        range_dropout_power=1.6,
        edge_dropout_strength=0.14,
        shadow_dropout_strength=0.16,
        minimum_observed_fraction=0.52,
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
        self.macro_relief_std = float(macro_relief_std)
        self.macro_relief_sigma = float(macro_relief_sigma)
        self.micro_relief_std = float(micro_relief_std)
        self.micro_relief_sigma = float(micro_relief_sigma)
        self.corrugation_amplitude = float(corrugation_amplitude)
        self.corrugation_wavelength = float(corrugation_wavelength)
        self.landing_zone_radius = float(landing_zone_radius)
        self.landing_zone_suppression = float(landing_zone_suppression)
        self.hazard_cluster_count = max(0, int(hazard_cluster_count))
        self.hazard_cluster_spread = float(hazard_cluster_spread)
        self.hazard_cluster_bias = float(hazard_cluster_bias)

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
        self.sensor_height = float(sensor_height)
        self.sensor_offset_x = float(sensor_offset_x)
        self.sensor_offset_y = float(sensor_offset_y)
        self.sensor_focus_gain = float(sensor_focus_gain)
        self.min_detection_probability = float(min_detection_probability)
        self.incidence_dropout_strength = float(incidence_dropout_strength)
        self.range_dropout_strength = float(range_dropout_strength)
        self.range_dropout_power = float(range_dropout_power)
        self.edge_dropout_strength = float(edge_dropout_strength)
        self.shadow_dropout_strength = float(shadow_dropout_strength)
        self.minimum_observed_fraction = float(minimum_observed_fraction)
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
        self.hazard_cluster_centers = self._build_hazard_cluster_centers()
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

        relief = np.zeros_like(dem)
        if self.macro_relief_std > 0:
            relief += gaussian_filter(
                self.rng.normal(0.0, self.macro_relief_std, size=xx.shape),
                sigma=max(self.macro_relief_sigma, 0.1),
            )
        if self.micro_relief_std > 0:
            relief += gaussian_filter(
                self.rng.normal(0.0, self.micro_relief_std, size=xx.shape),
                sigma=max(self.micro_relief_sigma, 0.05),
            )
        if self.corrugation_amplitude > 0 and self.corrugation_wavelength > 0:
            phase = self.rng.uniform(0.0, 2.0 * np.pi)
            corrugation = np.sin((2.0 * np.pi * yy / self.corrugation_wavelength) + phase)
            relief += self.corrugation_amplitude * corrugation

        if self.landing_zone_radius > 0 and self.landing_zone_suppression > 0:
            landing_weight = self._build_landing_zone_weight_map(xx, yy)
            relief *= 1.0 - np.clip(self.landing_zone_suppression * landing_weight, 0.0, 0.95)

        dem = dem + relief

        return dem, xx, yy

    def _build_landing_zone_weight_map(self, xx=None, yy=None):
        if xx is None or yy is None:
            xx = self.xx
            yy = self.yy
        if self.landing_zone_radius <= 0:
            return np.zeros_like(xx, dtype=float)
        radial_distance = np.sqrt(xx ** 2 + yy ** 2)
        normalized = radial_distance / max(self.landing_zone_radius, 1e-12)
        return np.exp(-(normalized ** 2))

    def _build_range_ratio_map(self):
        radial_distance = np.sqrt(self.xx ** 2 + self.yy ** 2)
        return radial_distance / max(np.max(radial_distance), 1e-12)

    def _estimate_surface_normals(self, dem):
        dz_dy, dz_dx = np.gradient(dem, self.pixel_size, self.pixel_size)
        normals = np.stack((-dz_dx, -dz_dy, np.ones_like(dem)), axis=-1)
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        return normals / np.maximum(norms, 1e-12)

    def _build_sensor_position(self, dem):
        terrain_reference = float(np.max(dem))
        return np.array(
            [
                self.sensor_offset_x,
                self.sensor_offset_y,
                terrain_reference + self.sensor_height,
            ],
            dtype=float,
        )

    def _compute_observation_probability_map(self, dem):
        points_grid = np.stack((self.xx, self.yy, dem), axis=-1)
        sensor_position = self._build_sensor_position(dem)
        vectors_to_sensor = sensor_position - points_grid
        ranges = np.linalg.norm(vectors_to_sensor, axis=-1)
        beam_dirs = vectors_to_sensor / np.maximum(ranges[..., None], 1e-12)

        normals = self._estimate_surface_normals(dem)
        incidence = np.clip(np.sum(normals * beam_dirs, axis=-1), 0.0, 1.0)

        range_ratio = (ranges - np.min(ranges)) / max(np.ptp(ranges), 1e-12)
        edge_ratio = np.sqrt(self.xx ** 2 + self.yy ** 2) / max(np.sqrt(2.0) * self.half_size, 1e-12)
        landing_weight = self._build_landing_zone_weight_map()

        sensor_xy_norm = np.linalg.norm(sensor_position[:2])
        if sensor_xy_norm <= 1e-12:
            sensor_xy_dir = np.zeros(2, dtype=float)
        else:
            sensor_xy_dir = sensor_position[:2] / sensor_xy_norm
        shadow_proxy = np.clip(
            -(normals[..., 0] * sensor_xy_dir[0] + normals[..., 1] * sensor_xy_dir[1]),
            0.0,
            1.0,
        )

        keep_probability = np.ones_like(dem, dtype=float)
        keep_probability *= 1.0 - self.incidence_dropout_strength * ((1.0 - incidence) ** 1.5)
        keep_probability *= 1.0 - self.range_dropout_strength * (range_ratio ** self.range_dropout_power)
        keep_probability *= 1.0 - self.edge_dropout_strength * (edge_ratio ** 1.8)
        keep_probability *= 1.0 - self.shadow_dropout_strength * (shadow_proxy ** 1.3)
        keep_probability += self.sensor_focus_gain * landing_weight
        keep_probability = np.clip(keep_probability, self.min_detection_probability, 1.0)

        return {
            "probability": keep_probability,
            "range_ratio": range_ratio,
            "incidence": incidence,
            "edge_ratio": edge_ratio,
            "shadow_proxy": shadow_proxy,
            "sensor_position": sensor_position,
        }

    def _sample_observation_mask(self, dem):
        observation = self._compute_observation_probability_map(dem)
        keep_probability = observation["probability"]
        observation_mask = self.rng.random(size=keep_probability.shape) < keep_probability

        minimum_points = max(3, int(round(self.total_points * self.minimum_observed_fraction)))
        observed_count = int(np.sum(observation_mask))
        if observed_count < minimum_points:
            flat_probs = keep_probability.ravel()
            top_indices = np.argsort(flat_probs)[-minimum_points:]
            observation_mask = np.zeros_like(flat_probs, dtype=bool)
            observation_mask[top_indices] = True
            observation_mask = observation_mask.reshape(keep_probability.shape)
            observed_count = int(np.sum(observation_mask))

        observation["mask"] = observation_mask
        observation["observed_count"] = observed_count
        observation["observed_fraction"] = observed_count / max(self.total_points, 1)
        observation["mean_keep_probability"] = float(np.mean(keep_probability))
        observation["mean_incidence"] = float(np.mean(observation["incidence"]))
        return observation

    def _save_json(self, payload, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _sample_uniform_outer_location(self, margin=0.0, exclusion_radius=None, max_attempts=200):
        valid_half_size = max(self.half_size - margin, self.pixel_size)
        exclusion_radius = max(float(exclusion_radius or 0.0), 0.0)

        for _ in range(max_attempts):
            x0 = self.rng.uniform(-valid_half_size, valid_half_size)
            y0 = self.rng.uniform(-valid_half_size, valid_half_size)
            if exclusion_radius <= 0 or np.hypot(x0, y0) >= exclusion_radius:
                return x0, y0

        x0 = self.rng.uniform(-valid_half_size, valid_half_size)
        y0 = self.rng.uniform(-valid_half_size, valid_half_size)
        return x0, y0

    def _build_hazard_cluster_centers(self):
        if self.hazard_cluster_count <= 0:
            return []

        exclusion_radius = max(self.landing_zone_radius * 0.9, self.pixel_size * 2.0)
        centers = []
        for _ in range(self.hazard_cluster_count):
            centers.append(
                self._sample_uniform_outer_location(
                    margin=self.pixel_size * 2.0,
                    exclusion_radius=exclusion_radius,
                )
            )
        return centers

    def _sample_hazard_location(self, margin):
        exclusion_radius = max(self.landing_zone_radius * 0.75, margin)
        valid_half_size = max(self.half_size - margin, self.pixel_size)

        if self.hazard_cluster_centers and self.rng.random() < self.hazard_cluster_bias:
            for _ in range(80):
                cx, cy = self.hazard_cluster_centers[int(self.rng.integers(0, len(self.hazard_cluster_centers)))]
                x0 = float(np.clip(
                    self.rng.normal(cx, self.hazard_cluster_spread),
                    -valid_half_size,
                    valid_half_size,
                ))
                y0 = float(np.clip(
                    self.rng.normal(cy, self.hazard_cluster_spread),
                    -valid_half_size,
                    valid_half_size,
                ))
                if np.hypot(x0, y0) >= exclusion_radius:
                    return x0, y0

        return self._sample_uniform_outer_location(margin=margin, exclusion_radius=exclusion_radius)

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

    @staticmethod
    def _segment_length_for_budget(primary_size, secondary_size, target_outliers):
        if target_outliers is None or target_outliers <= 0:
            return int(secondary_size)
        approx = int(np.ceil(float(target_outliers) / max(int(primary_size), 1)))
        return int(np.clip(approx, 1, int(secondary_size)))

    def _generate_stripe_artifact(self, dem, target_outliers=None):
        axis = self.stripe_axis if self.stripe_axis in {"x", "y"} else "y"
        width_min, width_max = self.stripe_width_range
        width = int(self.rng.integers(width_min, width_max + 1))
        signed_bias = self._random_signed_amplitude(self.stripe_bias_range)

        if axis == "x":
            start = int(self.rng.integers(0, max(self.cols - width + 1, 1)))
            segment_len = self._segment_length_for_budget(width, self.rows, target_outliers)
            segment_start = int(self.rng.integers(0, max(self.rows - segment_len + 1, 1)))
            mask = np.zeros_like(dem, dtype=bool)
            mask[segment_start : segment_start + segment_len, start : start + width] = True
        else:
            start = int(self.rng.integers(0, max(self.rows - width + 1, 1)))
            segment_len = self._segment_length_for_budget(width, self.cols, target_outliers)
            segment_start = int(self.rng.integers(0, max(self.cols - segment_len + 1, 1)))
            mask = np.zeros_like(dem, dtype=bool)
            mask[start : start + width, segment_start : segment_start + segment_len] = True

        stripe_texture = gaussian_filter(
            self.rng.normal(0.0, self.stripe_jitter_std, size=dem.shape),
            sigma=0.6,
        )
        dem[mask] += (signed_bias + stripe_texture[mask]) * (1.0 + 0.35 * self.range_ratio_map[mask])
        return mask

    def _generate_bad_batch(self, dem, target_outliers=None):
        axis = self.bad_batch_axis if self.bad_batch_axis in {"x", "y"} else "y"
        len_min, len_max = self.bad_batch_length_range
        batch_len = int(self.rng.integers(len_min, len_max + 1))
        signed_bias = self._random_signed_amplitude(self.bad_batch_bias_range)

        if axis == "x":
            start = int(self.rng.integers(0, max(self.cols - batch_len + 1, 1)))
            cols = np.arange(start, min(start + batch_len, self.cols))
            segment_len = self._segment_length_for_budget(len(cols), self.rows, target_outliers)
            segment_start = int(self.rng.integers(0, max(self.rows - segment_len + 1, 1)))
            rows = np.arange(segment_start, min(segment_start + segment_len, self.rows))
            phase = self.rng.uniform(0.0, 2.0 * np.pi)
            waveform = np.sin(np.linspace(0.0, 2.0 * np.pi, len(rows)) + phase)
            for offset, col in enumerate(cols):
                local_bias = signed_bias + self.rng.normal(0.0, self.bad_batch_jitter_std)
                dem[rows, col] += local_bias + 0.35 * self.bad_batch_jitter_std * waveform
            mask = np.zeros_like(dem, dtype=bool)
            mask[np.ix_(rows, cols)] = True
        else:
            start = int(self.rng.integers(0, max(self.rows - batch_len + 1, 1)))
            rows = np.arange(start, min(start + batch_len, self.rows))
            segment_len = self._segment_length_for_budget(len(rows), self.cols, target_outliers)
            segment_start = int(self.rng.integers(0, max(self.cols - segment_len + 1, 1)))
            cols = np.arange(segment_start, min(segment_start + segment_len, self.cols))
            phase = self.rng.uniform(0.0, 2.0 * np.pi)
            waveform = np.sin(np.linspace(0.0, 2.0 * np.pi, len(cols)) + phase)
            for offset, row in enumerate(rows):
                local_bias = signed_bias + self.rng.normal(0.0, self.bad_batch_jitter_std)
                dem[row, cols] += local_bias + 0.35 * self.bad_batch_jitter_std * waveform
            mask = np.zeros_like(dem, dtype=bool)
            mask[np.ix_(rows, cols)] = True

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
        observation = self._sample_observation_mask(dem)
        mask_flat = observation["mask"].flatten()

        x_nominal = self.xx.flatten()
        y_nominal = self.yy.flatten()
        z_nominal = dem.flatten()
        labels_flat = labels.flatten()
        range_ratio_flat = observation["range_ratio"].flatten()

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
        point_cloud = point_cloud[mask_flat]
        np.savetxt(output_path, point_cloud, fmt="%.4f %.4f %.4f %d")
        return {
            "observed_point_count": int(point_cloud.shape[0]),
            "observed_fraction": float(observation["observed_fraction"]),
            "mean_keep_probability": float(observation["mean_keep_probability"]),
            "mean_incidence": float(observation["mean_incidence"]),
            "sensor_position": [float(value) for value in observation["sensor_position"]],
            "observed_outlier_count": int(np.sum(point_cloud[:, 3] == 0)),
            "observed_inlier_count": int(np.sum(point_cloud[:, 3] == 1)),
        }

    def _apply_component_with_budget(self, component_name, budget, current_outliers, generator, max_attempts):
        if budget <= 0:
            return current_outliers, 0

        start_outliers = current_outliers
        target_total = min(self.total_points, start_outliers + int(budget))
        attempts = 0

        while current_outliers < target_total and attempts < max_attempts:
            attempts += 1
            dem_before = self.actual_dem.copy()
            generator(self.actual_dem, max(target_total - current_outliers, 1))
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
        ratio_factor = float(np.clip(target_outliers / 1000.0, 0.1, 0.9))
        min_dia = 0.5 + (ratio_factor - 0.1) * (2.0 - 0.5) / 0.8
        max_dia = 1.0 + (ratio_factor - 0.1) * (10.0 - 1.0) / 0.8
        diameter = self.rng.uniform(min_dia, max_dia)
        margin = diameter * 0.05
        x0, y0 = self._sample_hazard_location(margin)
        impact_angle = self.rng.uniform(40.0, 80.0)
        self._generate_crater(dem, x0, y0, diameter, impact_angle=impact_angle)

    def _sample_rock(self, dem, target_outliers):
        ratio_factor = float(np.clip(target_outliers / 1000.0, 0.1, 0.9))
        min_rad = 0.1 + (ratio_factor - 0.1) * (0.3 - 0.1) / 0.8
        max_rad = 0.5 + (ratio_factor - 0.1) * (0.8 - 0.5) / 0.8
        radius = self.rng.uniform(min_rad, max_rad)
        margin = radius * 0.3
        x0, y0 = self._sample_hazard_location(margin)
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
            lambda dem, remaining: self._sample_crater(dem, remaining),
            max_attempts=5000,
        )
        current_outliers, rock_added = self._apply_component_with_budget(
            "rocks",
            budgets["rock"],
            current_outliers,
            lambda dem, remaining: self._sample_rock(dem, remaining),
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
        actual_ratio = 100.0 * current_outliers / max(self.total_points, 1)

        dem_path, pc_path = build_ratio_output_paths(dem_dir, pc_dir, outlier_ratio)

        self._save_geotiff(self.actual_dem, dem_path)
        observation_stats = self._save_point_cloud(self.actual_dem, final_labels, pc_path)
        metadata_path = os.path.splitext(pc_path)[0] + "_metadata.json"
        observed_ratio = (
            100.0 * observation_stats["observed_outlier_count"] / max(observation_stats["observed_point_count"], 1)
        )
        self._save_json(
            {
                "random_state": int(self.random_state),
                "nominal_outlier_ratio": float(outlier_ratio),
                "actual_grid_outlier_ratio_percent": float(actual_ratio),
                "observed_outlier_ratio_percent": float(observed_ratio),
                "component_counts": {
                    "crater": int(crater_added),
                    "rock": int(rock_added),
                    "stripe": int(stripe_added),
                    "bad_batch": int(batch_added),
                    "random": int(random_added),
                },
                "sensor_model": {
                    "sensor_height": float(self.sensor_height),
                    "sensor_offset_x": float(self.sensor_offset_x),
                    "sensor_offset_y": float(self.sensor_offset_y),
                    "sensor_focus_gain": float(self.sensor_focus_gain),
                    "min_detection_probability": float(self.min_detection_probability),
                    "incidence_dropout_strength": float(self.incidence_dropout_strength),
                    "range_dropout_strength": float(self.range_dropout_strength),
                    "range_dropout_power": float(self.range_dropout_power),
                    "edge_dropout_strength": float(self.edge_dropout_strength),
                    "shadow_dropout_strength": float(self.shadow_dropout_strength),
                    "minimum_observed_fraction": float(self.minimum_observed_fraction),
                },
                "observation_stats": observation_stats,
            },
            metadata_path,
        )

        print(
            "  Final composition: "
            f"crater={crater_added}, rock={rock_added}, stripe={stripe_added}, "
            f"bad_batch={batch_added}, random={random_added}"
        )
        print(
            f"  Final result: {current_outliers} outliers ({actual_ratio:.1f}%), "
            f"observed points={observation_stats['observed_point_count']}"
        )
        return {
            "outlier_ratio": float(outlier_ratio),
            "actual_outlier_ratio_percent": float(actual_ratio),
            "outlier_count": int(current_outliers),
            "point_count": int(self.total_points),
            "observed_point_count": int(observation_stats["observed_point_count"]),
            "observed_outlier_ratio_percent": float(observed_ratio),
            "dem_path": dem_path,
            "point_cloud_path": pc_path,
            "metadata_path": metadata_path,
            "component_counts": {
                "crater": int(crater_added),
                "rock": int(rock_added),
                "stripe": int(stripe_added),
                "bad_batch": int(batch_added),
                "random": int(random_added),
            },
        }


def build_default_structured_simulator_kwargs(slope_angle=35.0, base_z=30.0, profile="landing", random_state=42):
    base_kwargs = {
        "size": 10,
        "slope_angle": slope_angle,
        "z": base_z,
        "threshold": 0.05,
        "z_noise_std": 0.08,
        "xy_noise_std": 0.05,
        "macro_relief_std": 0.035,
        "macro_relief_sigma": 1.8,
        "micro_relief_std": 0.012,
        "micro_relief_sigma": 0.40,
        "corrugation_amplitude": 0.018,
        "corrugation_wavelength": 3.2,
        "landing_zone_radius": 1.6,
        "landing_zone_suppression": 0.80,
        "hazard_cluster_count": 4,
        "hazard_cluster_spread": 1.05,
        "hazard_cluster_bias": 0.72,
        "crater_outlier_share": 0.40,
        "rock_outlier_share": 0.20,
        "stripe_outlier_share": 0.18,
        "bad_batch_outlier_share": 0.17,
        "range_noise_gain": 1.8,
        "range_xy_gain": 0.9,
        "sensor_height": 14.0,
        "sensor_offset_x": 0.8,
        "sensor_offset_y": -1.6,
        "sensor_focus_gain": 0.10,
        "min_detection_probability": 0.34,
        "incidence_dropout_strength": 0.40,
        "range_dropout_strength": 0.18,
        "range_dropout_power": 1.6,
        "edge_dropout_strength": 0.14,
        "shadow_dropout_strength": 0.16,
        "minimum_observed_fraction": 0.54,
        "random_state": int(random_state),
    }

    profile_name = str(profile).strip().lower()
    if profile_name == "baseline":
        base_kwargs.update({
            "macro_relief_std": 0.018,
            "micro_relief_std": 0.006,
            "corrugation_amplitude": 0.010,
            "landing_zone_radius": 0.0,
            "landing_zone_suppression": 0.0,
            "hazard_cluster_count": 0,
            "hazard_cluster_bias": 0.0,
            "crater_outlier_share": 0.45,
            "rock_outlier_share": 0.15,
            "sensor_focus_gain": 0.02,
            "min_detection_probability": 0.55,
            "incidence_dropout_strength": 0.10,
            "range_dropout_strength": 0.06,
            "edge_dropout_strength": 0.05,
            "shadow_dropout_strength": 0.05,
            "minimum_observed_fraction": 0.75,
        })
    elif profile_name == "stress":
        base_kwargs.update({
            "macro_relief_std": 0.050,
            "micro_relief_std": 0.018,
            "corrugation_amplitude": 0.028,
            "landing_zone_radius": 1.1,
            "landing_zone_suppression": 0.45,
            "hazard_cluster_count": 5,
            "hazard_cluster_spread": 1.35,
            "hazard_cluster_bias": 0.80,
            "stripe_outlier_share": 0.22,
            "bad_batch_outlier_share": 0.20,
            "range_noise_gain": 2.1,
            "range_xy_gain": 1.1,
            "sensor_focus_gain": 0.06,
            "min_detection_probability": 0.22,
            "incidence_dropout_strength": 0.55,
            "range_dropout_strength": 0.26,
            "edge_dropout_strength": 0.20,
            "shadow_dropout_strength": 0.24,
            "minimum_observed_fraction": 0.45,
        })
    return base_kwargs


def build_ratio_values(ratio_start=0.10, ratio_stop=0.91, ratio_step=0.01):
    return [float(round(value, 6)) for value in np.arange(ratio_start, ratio_stop, ratio_step)]


def outlier_ratio_to_percent(outlier_ratio):
    return int(round(float(outlier_ratio) * 100))


def build_dem_filename(outlier_ratio):
    return f"dem_{outlier_ratio_to_percent(outlier_ratio)}pct.tif"


def build_point_cloud_filename(outlier_ratio):
    return f"point_cloud_{outlier_ratio_to_percent(outlier_ratio)}pct.txt"


def build_ratio_output_paths(dem_dir, pc_dir, outlier_ratio):
    dem_path = os.path.join(dem_dir, build_dem_filename(outlier_ratio))
    pc_path = os.path.join(pc_dir, build_point_cloud_filename(outlier_ratio))
    return dem_path, pc_path


def build_ratio_seed(base_seed, ratio_index, outlier_ratio):
    return int(base_seed) + int(ratio_index) * 997 + outlier_ratio_to_percent(outlier_ratio) * 100


def simulation_outputs_exist(dem_dir, pc_dir, outlier_ratio):
    dem_path, pc_path = build_ratio_output_paths(dem_dir, pc_dir, outlier_ratio)
    npy_fallback = os.path.splitext(dem_path)[0] + ".npy"
    return os.path.exists(pc_path) and (os.path.exists(dem_path) or os.path.exists(npy_fallback))


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
    if simulator_kwargs.pop("_skip_existing", False) and simulation_outputs_exist(dem_dir, pc_dir, outlier_ratio):
        return {
            "outlier_ratio": float(outlier_ratio),
            "dem_path": build_ratio_output_paths(dem_dir, pc_dir, outlier_ratio)[0],
            "point_cloud_path": build_ratio_output_paths(dem_dir, pc_dir, outlier_ratio)[1],
            "skipped": True,
        }
    simulator = OutlierTerrainSimulator(**simulator_kwargs)
    return simulator.generate_simulation(outlier_ratio, dem_dir, pc_dir)


def generate_structured_dataset(
    dem_dir,
    pc_dir,
    ratio_start=0.10,
    ratio_stop=0.91,
    ratio_step=0.01,
    simulator_kwargs=None,
    max_workers=1,
    skip_existing=False,
):
    simulator_kwargs = dict(simulator_kwargs or {})
    os.makedirs(dem_dir, exist_ok=True)
    os.makedirs(pc_dir, exist_ok=True)

    ratio_values = build_ratio_values(ratio_start=ratio_start, ratio_stop=ratio_stop, ratio_step=ratio_step)
    if not ratio_values:
        print("No ratios scheduled for structured dataset generation.")
        return []

    max_workers = min(_resolve_max_workers(max_workers), len(ratio_values))
    base_seed = int(simulator_kwargs.get("random_state", 42))

    if max_workers <= 1:
        for idx, ratio in enumerate(tqdm(ratio_values, desc="Generating structured data")):
            worker_kwargs = dict(simulator_kwargs)
            worker_kwargs["random_state"] = build_ratio_seed(base_seed, idx, ratio)
            worker_kwargs["_skip_existing"] = skip_existing
            _simulate_ratio_worker(worker_kwargs, ratio, dem_dir, pc_dir)
    else:
        executor, backend = _build_parallel_executor(max_workers)
        print(f"Generating structured dataset with {max_workers} {backend} workers.")
        with executor:
            future_to_ratio = {}
            for idx, ratio in enumerate(ratio_values):
                worker_kwargs = dict(simulator_kwargs)
                worker_kwargs["random_state"] = build_ratio_seed(base_seed, idx, ratio)
                worker_kwargs["_skip_existing"] = skip_existing
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
    parser.add_argument(
        "--profile",
        default="landing",
        choices=["baseline", "landing", "stress"],
        help="Structured terrain profile. 'landing' is the recommended default for the paper workflow.",
    )
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--ratio-start", type=float, default=0.10)
    parser.add_argument("--ratio-stop", type=float, default=0.91)
    parser.add_argument("--ratio-step", type=float, default=0.01)
    parser.add_argument(
        "--workers",
        default=1,
        help="Number of worker processes for ratio-level parallel generation. Use 0 or 'auto' for CPU-count-based auto mode.",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Overwrite existing DEM and point-cloud files instead of skipping already generated ratios.",
    )
    return parser.parse_args()


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    args = parse_args()
    simulator_kwargs = build_default_structured_simulator_kwargs(
        slope_angle=args.slope_angle,
        base_z=args.base_z,
        profile=args.profile,
        random_state=args.base_seed,
    )
    generate_structured_dataset(
        dem_dir=args.dem_dir,
        pc_dir=args.pc_dir,
        ratio_start=args.ratio_start,
        ratio_stop=args.ratio_stop,
        ratio_step=args.ratio_step,
        simulator_kwargs=simulator_kwargs,
        max_workers=args.workers,
        skip_existing=not args.force_regenerate,
    )

    print(
        "\nStructured synthetic terrain generation finished:\n"
        f"DEM directory: {os.path.abspath(args.dem_dir)}\n"
        f"Point-cloud directory: {os.path.abspath(args.pc_dir)}"
    )


if __name__ == "__main__":
    main()
