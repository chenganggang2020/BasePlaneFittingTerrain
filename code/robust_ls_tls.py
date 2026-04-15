import numpy as np


class RobustLSTLSPlaneFitter:
    """
    Robust mixed LS-TLS plane fitting.

    The implementation follows the paper's practical strategy:
    1. Use a loose RANSAC pre-filter to suppress gross outliers.
    2. Estimate scale with MAD.
    3. Update Huber weights adaptively.
    4. Refit plane parameters with a weighted mixed TLS step.
    5. Update only points whose residuals changed significantly.
    """

    def __init__(
        self,
        ransac_tau=0.12,
        ransac_max_sample_num=2000,
        max_iter=50,
        tol=1e-5,
        huber_c=1.345,
        adaptive_c=True,
        min_c=0.5,
        sparse_eta=0.25,
        weight_decay=0.6,
        random_state=42,
    ):
        self.ransac_tau = ransac_tau
        self.ransac_max_sample_num = ransac_max_sample_num
        self.max_iter = max_iter
        self.tol = tol
        self.huber_c = huber_c
        self.adaptive_c = adaptive_c
        self.min_c = min_c
        self.sparse_eta = sparse_eta
        self.weight_decay = weight_decay
        self.random_state = random_state

        self.n_est = None
        self.d_est = None
        self.inliers = None

    @staticmethod
    def _plane_from_3points(points):
        """Estimate a plane from three non-collinear points."""
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        n_est = np.cross(v1, v2)
        if np.linalg.norm(n_est) < 1e-12:
            return None, None
        d_est = -float(np.dot(n_est, points[0]))
        return n_est, d_est

    @staticmethod
    def _orthogonal_residuals(points, n_est, d_est):
        """Compute point-to-plane orthogonal residuals."""
        denom = np.linalg.norm(n_est)
        if denom < 1e-12:
            return np.full(points.shape[0], np.inf, dtype=float)
        numer = np.abs(np.dot(points, n_est) + d_est)
        return numer / denom

    @staticmethod
    def _fit_plane_lsq(points):
        """
        Fit z = ax + by + c and convert it to the repository's plane form
        ax + by - z + d = 0.
        """
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        design = np.column_stack((x, y, np.ones(points.shape[0])))
        params, _, _, _ = np.linalg.lstsq(design, z, rcond=None)
        a_est, b_est, c_est = params
        n_est = np.array([a_est, b_est, -1.0], dtype=float)
        d_est = float(c_est)
        if d_est < 0:
            n_est = -n_est
            d_est = -d_est
        return n_est, d_est

    def _simple_ransac_prefilter(self, points):
        """Return a candidate inlier mask using a loose RANSAC pass."""
        n_total = points.shape[0]
        if n_total < 3:
            raise ValueError("Need at least 3 points for plane fitting")

        rng = np.random.default_rng(self.random_state)
        best_count = 0
        best_mask = np.ones(n_total, dtype=bool)

        for _ in range(self.ransac_max_sample_num):
            sample_idx = rng.choice(n_total, 3, replace=False)
            n_curr, d_curr = self._plane_from_3points(points[sample_idx])
            if n_curr is None or np.linalg.norm(n_curr) < 1e-12:
                continue

            residuals = self._orthogonal_residuals(points, n_curr, d_curr)
            mask = residuals < self.ransac_tau
            count = int(np.sum(mask))
            if count > best_count:
                best_count = count
                best_mask = mask

        if best_count >= 3:
            return best_mask
        return np.ones(n_total, dtype=bool)

    @staticmethod
    def _mad_sigma(residuals):
        residuals = np.asarray(residuals, dtype=float)
        centered = residuals - np.median(residuals)
        mad = np.median(np.abs(centered))
        return max(1.4826 * mad, 1e-8)

    def _adaptive_huber_c(self, residuals, sigma):
        if not self.adaptive_c:
            return self.huber_c
        if sigma <= 0:
            return self.huber_c
        q95 = np.quantile(np.abs(residuals), 0.95)
        return max(self.min_c, q95 / sigma)

    @staticmethod
    def _huber_weights(residuals, sigma, c_value):
        abs_res = np.abs(residuals)
        threshold = c_value * sigma
        weights = np.ones_like(abs_res, dtype=float)
        mask = abs_res > threshold
        if np.any(mask):
            weights[mask] = threshold / np.maximum(abs_res[mask], 1e-12)
        return np.clip(weights, 1e-6, 1.0)

    @staticmethod
    def _weighted_mixed_tls(points, weights):
        """
        Solve the mixed LS-TLS step by weighted centered SVD on [x, y, z],
        then recover the intercept with weighted LS.
        """
        weights = np.asarray(weights, dtype=float).reshape(-1)
        weights = np.clip(weights, 1e-8, None)
        weights = weights / np.sum(weights)

        centroid = np.sum(points * weights[:, None], axis=0)
        centered = points - centroid
        weighted_centered = centered * np.sqrt(weights[:, None])

        _, _, vh = np.linalg.svd(weighted_centered, full_matrices=False)
        normal = vh[-1]

        if abs(normal[2]) < 1e-10:
            normal[2] = np.sign(normal[2]) * 1e-10 if normal[2] != 0 else 1e-10

        a = -normal[0] / normal[2]
        b = -normal[1] / normal[2]

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        c = np.sum(weights * (z - a * x - b * y))

        n_est = np.array([a, b, -1.0], dtype=float)
        d_est = c

        if d_est < 0:
            n_est = -n_est
            d_est = -d_est

        return n_est, d_est

    def fit(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")
        if points.shape[0] < 3:
            raise ValueError("Need at least 3 points for plane fitting")

        prefilter_mask = self._simple_ransac_prefilter(points)
        candidate_points = points[prefilter_mask]
        if candidate_points.shape[0] < 3:
            candidate_points = points

        try:
            n_curr, d_curr = self._fit_plane_lsq(candidate_points)
        except Exception:
            n_curr, d_curr = self._weighted_mixed_tls(candidate_points, np.ones(candidate_points.shape[0]))

        prev_residuals = None
        weights = np.ones(candidate_points.shape[0], dtype=float)

        for _ in range(self.max_iter):
            prev_weights = weights.copy()
            vertical_residuals = candidate_points[:, 2] - (
                n_curr[0] * candidate_points[:, 0] + n_curr[1] * candidate_points[:, 1] + d_curr
            )
            sigma = self._mad_sigma(vertical_residuals)
            c_value = self._adaptive_huber_c(vertical_residuals, sigma)
            new_weights = self._huber_weights(vertical_residuals, sigma, c_value)

            if prev_residuals is None:
                update_mask = np.ones_like(new_weights, dtype=bool)
            else:
                update_mask = np.abs(vertical_residuals - prev_residuals) > (self.sparse_eta * sigma)

            if np.any(update_mask):
                weights[update_mask] = (
                    self.weight_decay * weights[update_mask]
                    + (1.0 - self.weight_decay) * new_weights[update_mask]
                )
            else:
                weights = self.weight_decay * weights + (1.0 - self.weight_decay) * new_weights

            weights = np.clip(weights, 1e-6, 1.0)
            n_next, d_next = self._weighted_mixed_tls(candidate_points, weights)

            param_delta = np.linalg.norm(n_next - n_curr) + abs(d_next - d_curr)
            weight_delta = np.max(np.abs(weights - prev_weights))

            n_curr, d_curr = n_next, d_next
            prev_residuals = vertical_residuals.copy()

            if param_delta < self.tol and weight_delta < self.tol:
                break

        final_residuals = self._orthogonal_residuals(points, n_curr, d_curr)
        final_sigma = self._mad_sigma(final_residuals)
        self.inliers = final_residuals < (2.5 * final_sigma)
        self.n_est = n_curr
        self.d_est = d_curr
        return self.n_est, self.d_est, self.inliers
