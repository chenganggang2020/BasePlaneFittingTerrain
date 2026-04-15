import numpy as np
from utils import plane_from_3points, compute_residuals, fit_plane_lsq


class RANSACPlaneFitter:
    def __init__(self, tau, p=0.99, eps=0.5, max_sample_num=1000):
        """
        初始化RANSAC参数
        :param tau: 残差阈值
        :param p: 成功概率
        :param eps: 初始外点比例估计
        :param max_sample_num: 最大采样次数
        """
        self.tau = tau
        self.p = p
        self.eps = eps
        self.max_sample_num = max_sample_num
        self.best_n = None
        self.best_d = None
        self.inliers = None

    def fit(self, points):
        """
        执行RANSAC平面拟合
        :param points: Nx3点云
        """
        n_total = points.shape[0]
        if n_total < 3:
            raise ValueError("点云数量需≥3")

        eps = self.eps
        if not (0 < eps < 1):
            eps = 0.5

        term = 1 - (1 - eps) ** 3
        if term <= 0:
            sample_num = self.max_sample_num
        else:
            sample_num = int(np.ceil(np.log(1 - self.p) / np.log(term)))
        sample_num = min(sample_num, self.max_sample_num)

        best_inlier_count = 0
        best_n, best_d = None, None

        for _ in range(sample_num):
            sample_idx = np.random.choice(n_total, 3, replace=False)
            n_curr, d_curr = plane_from_3points(points[sample_idx])
            if n_curr is None:
                continue

            residuals = compute_residuals(points, n_curr, d_curr)
            inliers_mask = residuals < self.tau
            inlier_count = np.sum(inliers_mask)

            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_n, best_d = n_curr, d_curr
                if best_inlier_count / n_total > (1 - eps) - 1e-6:
                    break

        if best_n is None:
            raise RuntimeError("未找到有效平面")

        residuals = compute_residuals(points, best_n, best_d)
        inliers_mask = abs(residuals) < self.tau
        if np.sum(inliers_mask) >= 3:
            best_n, best_d = fit_plane_lsq(points[inliers_mask])

        self.best_n = best_n
        self.best_d = best_d
        if self.best_d < 0:
            self.best_n = -self.best_n
            self.best_d = -self.best_d
        self.inliers = inliers_mask
        return  self.best_n,  self.best_d,  self.inliers
