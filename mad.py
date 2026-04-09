import numpy as np
from utils import plane_from_3points, compute_residuals, fit_plane_lsq


class MADPlaneFitter:
    def __init__(self, sigma_factor=1.4826, tol=1e-3, max_iter=100):
        """
        初始化MAD平面拟合类
        :param sigma_factor: 校正因子（默认1.4826）
        :param tol: 收敛阈值
        :param max_iter: 最大迭代次数
        """
        self.sigma_factor = sigma_factor
        self.tol = tol
        self.max_iter = max_iter
        self.n = None
        self.d = None
        self.inliers = None

    def fit(self, points):
        """
        执行MAD平面拟合
        :param points: Nx3点云数组
        :return: n, d, inliers_mask
        """
        n_total = points.shape[0]
        if n_total < 3:
            raise ValueError("点云数量需≥3")

        # 初始化平面
        n_prev, d_prev = None, None
        while n_prev is None:
            sample_idx = np.random.choice(n_total, 3, replace=False)
            n_prev, d_prev = plane_from_3points(points[sample_idx])

        # 迭代优化
        for _ in range(self.max_iter):
            residuals = compute_residuals(points, n_prev, d_prev)
            med_res = np.median(residuals)
            mad_scale = self.sigma_factor * med_res
            inliers_mask = residuals < 2.5 * mad_scale

            if np.sum(inliers_mask) < 3:
                # 内点不足，再随机抽样
                sample_idx = np.random.choice(n_total, 3, replace=False)
                n_prev, d_prev = plane_from_3points(points[sample_idx])
                continue

            n_curr, d_curr = fit_plane_lsq(points[inliers_mask])
            # 收敛判定
            if np.linalg.norm(n_curr - n_prev) < self.tol:
                break
            n_prev, d_prev = n_curr, d_curr

        # 最终筛选内点
        final_residuals = compute_residuals(points, n_prev, d_prev)
        med_res_final = np.median(final_residuals)
        mad_scale_final = self.sigma_factor * med_res_final
        inliers_mask = final_residuals < 2.5 * mad_scale_final

        self.n = n_prev
        self.d = d_prev
        if self.d < 0:
            self.n = -self.n
            self.d = -self.d
        self.inliers = inliers_mask
        return self.n, self.d, self.inliers
