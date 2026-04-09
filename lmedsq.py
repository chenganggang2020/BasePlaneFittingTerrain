import numpy as np
from utils import plane_from_3points, compute_residuals, fit_plane_lsq


class LMedsqPlaneFitter:
    def __init__(self, sample_num=1000, tol=1e-3):
        """
        初始化LMedsq平面拟合类
        :param sample_num: 采样次数
        :param tol: 残差平方中位数比较阈值
        """
        self.sample_num = sample_num
        self.tol = tol
        self.best_n = None
        self.best_d = None
        self.inliers = None

    def fit(self, points):
        """
        执行LMedsq平面拟合
        :param points: Nx3点云数组
        :return: n, d, inliers_mask
        """
        n_total = points.shape[0]
        if n_total < 3:
            raise ValueError("点云数量需≥3")

        best_med_sq_res = np.inf
        best_n, best_d = None, None

        for _ in range(self.sample_num):
            sample_idx = np.random.choice(n_total, 3, replace=False)
            n_curr, d_curr = plane_from_3points(points[sample_idx])
            if n_curr is None:
                continue

            sq_residuals = compute_residuals(points, n_curr, d_curr) ** 2
            med_sq_res = np.median(sq_residuals)

            if med_sq_res < best_med_sq_res - self.tol:
                best_med_sq_res = med_sq_res
                best_n, best_d = n_curr, d_curr

        if best_n is None:
            raise RuntimeError("未找到有效候选平面")

        residuals = compute_residuals(points, best_n, best_d)
        threshold = 1.4826 * np.sqrt(best_med_sq_res)
        inliers_mask = residuals < threshold

        if np.sum(inliers_mask) >= 3:
            best_n, best_d = fit_plane_lsq(points[inliers_mask])

        self.best_n = best_n
        self.best_d = best_d
        if self.best_d < 0:
            self.best_n = -self.best_n
            self.best_d = -self.best_d
        self.inliers = inliers_mask
        return  self.best_n,  self.best_d,  self.inliers
