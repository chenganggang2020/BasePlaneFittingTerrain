import numpy as np


class SequentialRobustTLSPlaneFitter:
    """
    Practical recursive PEIV / robust TLS plane fitting.

    Compared with the previous batch-style prototype, this implementation follows
    the thesis workflow more closely:
    1. Use a RANSAC-based initializer.
    2. Split the point cloud into groups.
    3. At every outer iteration, build observation-space and structure-space
       robust weights with an IGGIII-style dual-factor scheme.
    4. Sequentially accumulate each group in information form to avoid a single
       global high-dimensional solve.
    """

    def __init__(
        self,
        sigma_x=0.03,
        sigma_y=0.03,
        sigma_z=0.05,
        k1=2.5,
        k2=4.5,
        batch_k1=1.8,
        batch_k2=3.2,
        max_iter=20,
        eps_xi=1e-8,
        batch_size=128,
        ransac_tau=0.12,
        ransac_max_sample_num=1000,
        prior_precision=1e-6,
        hard_threshold_scale=2.0,
        hard_sigma_scale=2.5,
        hard_trim_quantile=0.82,
        hard_refit_iters=2,
        min_inlier_ratio=0.20,
        spatial_sort=True,
        random_state=42,
    ):
        self.sigma_x = float(sigma_x)
        self.sigma_y = float(sigma_y)
        self.sigma_z = float(sigma_z)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.batch_k1 = float(batch_k1)
        self.batch_k2 = float(batch_k2)
        self.max_iter = int(max_iter)
        self.eps_xi = float(eps_xi)
        self.batch_size = int(batch_size)
        self.ransac_tau = float(ransac_tau)
        self.ransac_max_sample_num = int(ransac_max_sample_num)
        self.prior_precision = float(prior_precision)
        self.hard_threshold_scale = float(hard_threshold_scale)
        self.hard_sigma_scale = float(hard_sigma_scale)
        self.hard_trim_quantile = float(hard_trim_quantile)
        self.hard_refit_iters = int(hard_refit_iters)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.spatial_sort = bool(spatial_sort)
        self.random_state = random_state

        self.n_est = None
        self.d_est = None
        self.inliers = None
        self.batch_states = []
        self.batch_reliabilities = []

    @staticmethod
    def _plane_from_3points(points):
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        n_est = np.cross(v1, v2)
        if np.linalg.norm(n_est) < 1e-12:
            return None, None
        d_est = -float(np.dot(n_est, points[0]))
        return n_est, d_est

    @staticmethod
    def _orthogonal_residuals(points, n_est, d_est):
        denom = np.linalg.norm(n_est)
        if denom < 1e-12:
            return np.full(points.shape[0], np.inf, dtype=float)
        numer = np.abs(np.dot(points, n_est) + d_est)
        return numer / denom

    @staticmethod
    def _mad_scale(values):
        values = np.asarray(values, dtype=float).reshape(-1)
        centered = values - np.median(values)
        mad = np.median(np.abs(centered))
        return max(1.4826 * mad, 1e-8)

    @staticmethod
    def _fit_plane_lsq(points):
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        design = np.column_stack((x, y, np.ones(points.shape[0])))
        params, _, _, _ = np.linalg.lstsq(design, z, rcond=None)
        return params.astype(float)

    @staticmethod
    def _state_to_plane(state, centroid=None):
        a_est, b_est, c_est = np.asarray(state, dtype=float).reshape(-1)
        if centroid is not None:
            c_est = c_est + centroid[2] - a_est * centroid[0] - b_est * centroid[1]

        n_est = np.array([a_est, b_est, -1.0], dtype=float)
        d_est = float(c_est)
        if d_est < 0:
            n_est = -n_est
            d_est = -d_est
        return n_est, d_est

    @staticmethod
    def _single_igg3_weight(value, k1, k2):
        value = abs(float(value))
        if value <= k1:
            return 1.0
        if value > k2:
            return 1e-8
        return (k1 / max(value, 1e-12)) * ((k2 - value) / max(k2 - k1, 1e-12)) ** 2

    def _batch_reliability(self, standardized_residuals):
        if len(standardized_residuals) == 0:
            return 1.0
        robust_level = np.quantile(np.abs(standardized_residuals), 0.75)
        return self._single_igg3_weight(robust_level, self.batch_k1, self.batch_k2)

    def _refit_on_mask(self, points, mask):
        mask = np.asarray(mask, dtype=bool)
        if np.sum(mask) < 3:
            return None, None
        params = self._fit_plane_lsq(points[mask])
        return self._state_to_plane(params, centroid=None)

    def _hard_trim_refit(self, points, initial_n_est, initial_d_est, model_sigma, combined_weights=None):
        n_est = np.asarray(initial_n_est, dtype=float)
        d_est = float(initial_d_est)
        residuals = self._orthogonal_residuals(points, n_est, d_est)
        min_keep = max(3, int(np.ceil(points.shape[0] * self.min_inlier_ratio)))

        weight_seed = None
        if combined_weights is not None:
            combined_weights = np.asarray(combined_weights, dtype=float).reshape(-1)
            quantile_level = min(max(1.0 - self.min_inlier_ratio, 0.50), 0.90)
            weight_gate = np.quantile(combined_weights, quantile_level)
            weight_seed = combined_weights >= weight_gate

        final_mask = None
        for _ in range(max(self.hard_refit_iters, 1)):
            residual_scale = self._mad_scale(residuals)
            threshold = max(
                self.hard_threshold_scale * residual_scale,
                self.hard_sigma_scale * model_sigma,
            )
            candidate_mask = residuals <= threshold

            if weight_seed is not None:
                candidate_mask = candidate_mask & weight_seed
                if np.sum(candidate_mask) < min_keep:
                    candidate_mask = residuals <= threshold

            if np.sum(candidate_mask) < min_keep:
                if points.shape[0] >= min_keep:
                    keep_idx = np.argsort(residuals)[:min_keep]
                    candidate_mask = np.zeros(points.shape[0], dtype=bool)
                    candidate_mask[keep_idx] = True
                else:
                    candidate_mask = np.ones(points.shape[0], dtype=bool)

            refit_n_est, refit_d_est = self._refit_on_mask(points, candidate_mask)
            if refit_n_est is None:
                final_mask = candidate_mask
                break

            n_est = refit_n_est
            d_est = refit_d_est
            residuals = self._orthogonal_residuals(points, n_est, d_est)
            final_mask = candidate_mask

            if np.sum(candidate_mask) < 6:
                break

            inlier_residuals = residuals[candidate_mask]
            adaptive_cap = np.quantile(
                inlier_residuals,
                min(max(self.hard_trim_quantile, 0.55), 0.98),
            )
            tightened_threshold = max(adaptive_cap, self.hard_sigma_scale * model_sigma)
            final_mask = residuals <= tightened_threshold

        if final_mask is None:
            final_mask = residuals <= max(
                self.hard_threshold_scale * self._mad_scale(residuals),
                self.hard_sigma_scale * model_sigma,
            )

        refit_n_est, refit_d_est = self._refit_on_mask(points, final_mask)
        if refit_n_est is not None:
            n_est = refit_n_est
            d_est = refit_d_est
            residuals = self._orthogonal_residuals(points, n_est, d_est)

        return n_est, d_est, final_mask, residuals

    def _simple_ransac_init(self, points):
        n_total = points.shape[0]
        if n_total < 3:
            raise ValueError("Need at least 3 points for plane fitting")

        rng = np.random.default_rng(self.random_state)
        best_count = 0
        best_mask = np.ones(n_total, dtype=bool)

        for _ in range(self.ransac_max_sample_num):
            sample_idx = rng.choice(n_total, 3, replace=False)
            n_curr, d_curr = self._plane_from_3points(points[sample_idx])
            if n_curr is None:
                continue

            residuals = self._orthogonal_residuals(points, n_curr, d_curr)
            mask = residuals < self.ransac_tau
            count = int(np.sum(mask))
            if count > best_count:
                best_count = count
                best_mask = mask

        candidate_points = points[best_mask] if best_count >= 3 else points
        return self._fit_plane_lsq(candidate_points)

    def _build_batches(self, points):
        n_points = points.shape[0]
        batch_size = max(16, min(self.batch_size, n_points))
        order = np.arange(n_points)
        if self.spatial_sort:
            order = np.lexsort((points[:, 1], points[:, 0]))
        return [order[start : start + batch_size] for start in range(0, n_points, batch_size)]

    def _split_peiv_residuals(self, residuals, state):
        a_est, b_est, _ = state
        sigma_obs_sq = self.sigma_z ** 2
        sigma_ax_sq = (a_est * self.sigma_x) ** 2
        sigma_by_sq = (b_est * self.sigma_y) ** 2
        total_sigma_sq = max(sigma_obs_sq + sigma_ax_sq + sigma_by_sq, 1e-10)

        e_l = residuals * sigma_obs_sq / total_sigma_sq
        e_ax = -residuals * sigma_ax_sq / total_sigma_sq
        e_ay = -residuals * sigma_by_sq / total_sigma_sq
        return e_l, e_ax, e_ay

    def _igg3_weights(self, standardized_residuals):
        standardized_residuals = np.abs(np.asarray(standardized_residuals, dtype=float))
        weights = np.ones_like(standardized_residuals)

        mask_down = (standardized_residuals > self.k1) & (standardized_residuals <= self.k2)
        if np.any(mask_down):
            values = standardized_residuals[mask_down]
            weights[mask_down] = (
                self.k1 / np.maximum(values, 1e-12)
            ) * ((self.k2 - values) / max(self.k2 - self.k1, 1e-12)) ** 2
        weights[standardized_residuals > self.k2] = 1e-8
        return np.clip(weights, 1e-8, 1.0)

    def _compute_dual_weights(self, points, state):
        design = np.column_stack((points[:, 0], points[:, 1], np.ones(points.shape[0])))
        predicted = design @ state
        vertical_residuals = points[:, 2] - predicted

        e_l, e_ax, e_ay = self._split_peiv_residuals(vertical_residuals, state)

        obs_std = np.abs(e_l) / max(self.sigma_z, 1e-8)
        struct_std = np.sqrt(
            (e_ax / max(self.sigma_x, 1e-8)) ** 2 +
            (e_ay / max(self.sigma_y, 1e-8)) ** 2
        )

        obs_scale = self._mad_scale(obs_std)
        struct_scale = self._mad_scale(struct_std)

        obs_weights = self._igg3_weights(obs_std / obs_scale)
        struct_weights = self._igg3_weights(struct_std / struct_scale)

        a_est, b_est, _ = state
        sigma_struct_sq = (a_est ** 2) * (self.sigma_x ** 2) + (b_est ** 2) * (self.sigma_y ** 2)
        equivalent_variance = (
            (self.sigma_z ** 2) / obs_weights +
            sigma_struct_sq / struct_weights
        )
        equivalent_variance = np.clip(equivalent_variance, 1e-10, None)

        combined_weights = np.sqrt(obs_weights * struct_weights)
        return {
            "vertical_residuals": vertical_residuals,
            "obs_weights": obs_weights,
            "struct_weights": struct_weights,
            "combined_weights": combined_weights,
            "equivalent_variance": equivalent_variance,
        }

    def _sequential_information_update(self, points, initial_state, weights_info):
        design = np.column_stack((points[:, 0], points[:, 1], np.ones(points.shape[0])))
        observations = points[:, 2]
        standardized_residuals = np.abs(weights_info["vertical_residuals"]) / np.sqrt(
            np.maximum(weights_info["equivalent_variance"], 1e-10)
        )

        information = np.eye(3, dtype=float) * self.prior_precision
        normal_vector = information @ initial_state.reshape(3, 1)
        state = initial_state.copy()
        batch_states = []
        batch_reliabilities = []

        for batch_indices in self._build_batches(points):
            h_batch = design[batch_indices]
            z_batch = observations[batch_indices]
            batch_factor = self._batch_reliability(standardized_residuals[batch_indices])
            weight_batch = batch_factor / weights_info["equivalent_variance"][batch_indices]

            information += h_batch.T @ (h_batch * weight_batch[:, None])
            normal_vector += h_batch.T @ (weight_batch * z_batch)[:, None]

            regularized_info = information + np.eye(3, dtype=float) * 1e-12
            try:
                state = np.linalg.solve(regularized_info, normal_vector).reshape(-1)
            except np.linalg.LinAlgError:
                state = np.linalg.lstsq(regularized_info, normal_vector, rcond=None)[0].reshape(-1)
            batch_states.append(state.copy())
            batch_reliabilities.append(batch_factor)

        return state, batch_states, batch_reliabilities

    def fit(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")
        if points.shape[0] < 3:
            raise ValueError("Need at least 3 points for plane fitting")

        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        state = self._simple_ransac_init(centered_points)

        prev_combined_weights = None
        self.batch_states = []
        self.batch_reliabilities = []

        for _ in range(self.max_iter):
            weight_info = self._compute_dual_weights(centered_points, state)
            next_state, batch_states, batch_reliabilities = self._sequential_information_update(
                centered_points,
                state,
                weight_info,
            )

            parameter_delta = np.linalg.norm(next_state - state)
            combined_weights = weight_info["combined_weights"]
            if prev_combined_weights is None:
                weight_delta = np.inf
            else:
                weight_delta = np.max(np.abs(combined_weights - prev_combined_weights))

            state = next_state
            prev_combined_weights = combined_weights
            self.batch_states = batch_states
            self.batch_reliabilities = batch_reliabilities

            if parameter_delta < self.eps_xi and weight_delta < 1e-3:
                break

        n_est, d_est = self._state_to_plane(state, centroid=centroid)
        a_est, b_est, _ = state

        orthogonal_residuals = self._orthogonal_residuals(points, n_est, d_est)
        model_sigma = np.sqrt(
            self.sigma_z ** 2 +
            (a_est ** 2) * (self.sigma_x ** 2) +
            (b_est ** 2) * (self.sigma_y ** 2)
        )
        n_est, d_est, inliers, orthogonal_residuals = self._hard_trim_refit(
            points,
            n_est,
            d_est,
            model_sigma,
            combined_weights=prev_combined_weights,
        )

        self.n_est = n_est
        self.d_est = d_est
        self.inliers = np.asarray(inliers, dtype=bool)
        return self.n_est, self.d_est, self.inliers
