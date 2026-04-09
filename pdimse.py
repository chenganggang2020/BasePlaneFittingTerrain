# pdimse.py

import numpy as np
from scipy.stats import chi2, norm
from utils import plane_from_3points, compute_residuals, fit_plane_lsq


class PDIMSEPlaneFitter:
    """
    PDIMSE（Preprocessing Dual Iterative Median Scale Estimator）平面拟合器
    """

    def __init__(self, alpha=0.95, P=0.95, eps=0.5, K_ratio=0.5, tol=1e-3, max_iter=100):
        """
        初始化 PDIMSE 参数

        :param alpha: 预处理卡方置信度（默认 0.95）
        :param P: 采样成功概率（默认 0.95）
        :param eps: 初始外点比例估计（默认 0.5）
        :param K_ratio: DIMSE 中 K 与内点数的比例（默认 0.5）
        :param tol: 尺度估计收敛阈值（默认 1e-3）
        :param max_iter: DIMSE 最大迭代次数（默认 100）
        """
        self.alpha = alpha
        self.P = P
        self.eps = eps
        self.K_ratio = K_ratio
        self.tol = tol
        self.max_iter = max_iter
        self.best_d=None
        self.best_n=None
        self.inliers = None

    def fit(self, points):
        """
        拟合平面并返回结果

        :param points: (N, 3) 点云数组
        :return: n (3,), d (float), inliers (bool array of shape (N,))
        """
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points 必须是 (N, 3) 的二维数组")

        n_total = points.shape[0]
        if n_total < 3:
            raise ValueError("点云数量需 ≥ 3")

        chi2_inv = chi2.ppf(self.alpha, df=1)

        # ===== 预处理：MSAC 风格保守内点筛选 =====
        term = 1 - (1 - self.eps) ** 3
        if term <= 0:
            sample_num = 10000
        else:
            sample_num = int(np.ceil(np.log(1 - self.P) / np.log(term)))
        sample_num = min(sample_num, 10000)

        best_cost = np.inf
        best_n_pre, best_d_pre = None, None
        best_residuals_pre = None

        for _ in range(sample_num):
            sample_idx = np.random.choice(n_total, 3, replace=False)
            n_curr, d_curr = plane_from_3points(points[sample_idx])
            if n_curr is None:
                continue

            residuals = compute_residuals(points, n_curr, d_curr)
            sigma_max = np.std(residuals)
            tau_max_sq = chi2_inv * (sigma_max ** 2)
            cost = np.sum(np.where(residuals ** 2 < tau_max_sq, residuals ** 2, tau_max_sq))

            if cost < best_cost:
                best_cost = cost
                best_n_pre, best_d_pre = n_curr, d_curr
                best_residuals_pre = residuals

        if best_n_pre is None:
            raise RuntimeError("预处理阶段未能找到有效平面")

        sigma_max_pre = np.std(best_residuals_pre)
        tau_max_pre = np.sqrt(chi2_inv * (sigma_max_pre ** 2))
        conservative_inliers = np.abs(best_residuals_pre) < tau_max_pre  # 注意：取绝对值！

        if np.sum(conservative_inliers) < 3:
            raise RuntimeError("预处理后保守内点不足 3 个")

        conservative_points = points[conservative_inliers]

        # ===== 双迭代中值尺度估计（DIMSE）=====
        m_in = np.sum(conservative_inliers)
        K = int(np.ceil(self.K_ratio * m_in))
        K = max(1, min(K, m_in))

        # 初始 LS 拟合
        n_prev, d_prev = fit_plane_lsq(conservative_points)
        residuals_prev = compute_residuals(conservative_points, n_prev, d_prev)
        r_med_prev = np.median(np.abs(residuals_prev))
        prob_prev = 0.5 * (1 + K / m_in)
        prob_prev = min(max(prob_prev, 1e-6), 1 - 1e-6)
        phi_inv_prev = norm.ppf(prob_prev)
        if phi_inv_prev < 1e-8:
            phi_inv_prev = 1e-8
        S_prev = r_med_prev / phi_inv_prev

        inliers_prev = None

        for _ in range(self.max_iter):
            inliers_curr = np.abs(residuals_prev) < 2.5 * S_prev
            n_inlier_curr = np.sum(inliers_curr)
            if n_inlier_curr < 3:
                break

            r_med_curr = np.median(np.abs(residuals_prev[inliers_curr]))
            prob_curr = 0.5 * (1 + K / n_inlier_curr)
            prob_curr = min(max(prob_curr, 1e-6), 1 - 1e-6)
            phi_inv_curr = norm.ppf(prob_curr)
            if phi_inv_curr < 1e-8:
                phi_inv_curr = 1e-8
            S_curr = r_med_curr / phi_inv_curr

            inlier_points = conservative_points[inliers_curr]
            n_curr, d_curr = fit_plane_lsq(inlier_points)
            residuals_curr = compute_residuals(conservative_points, n_curr, d_curr)

            if np.abs(S_curr - S_prev) / (S_prev + 1e-12) < self.tol:
                S_prev = S_curr
                n_prev, d_prev = n_curr, d_curr
                inliers_prev = inliers_curr
                break

            S_prev = S_curr
            n_prev, d_prev = n_curr, d_curr
            residuals_prev = residuals_curr
            inliers_prev = inliers_curr

        if d_prev < 0:
            self.best_n = -n_prev
            self.best_d = -d_prev
        else :
            self.best_n = n_prev
            self.best_d = d_prev
        # ===== 构建全局内点掩码 =====
        conservative_idx = np.where(conservative_inliers)[0]
        if inliers_prev is None:
            inliers_prev = np.ones_like(residuals_prev, dtype=bool)

        final_inliers_conservative = np.abs(residuals_prev) < 2.5 * S_prev
        final_inliers = np.zeros(n_total, dtype=bool)
        final_inliers[conservative_idx[final_inliers_conservative]] = True

        # 如果最终内点足够，用它们重新拟合一次
        if np.sum(final_inliers) >= 3:
            n_prev, d_prev = fit_plane_lsq(points[final_inliers])
        self.inliers=final_inliers
        return  self.best_n,  self.best_d,  self.inliers