import numpy as np
from scipy.stats import norm
from utils import fit_plane_lsq, compute_residuals


class IKOSEPlaneFitter:
    """
    原始 IKOSE (Wang et al., 2012) 实现
    特征：
    1. 无 RANSAC 预处理 (直接从全局 LSQ 开始)
    2. 固定锚点 (Fixed Anchor): 迭代中分子 r_K 保持不变
    """

    def __init__(self, K_ratio=0.1, tol=1e-4, max_iter=20):
        # 注意：原始文献中 K 通常取 10% (0.1) 以保证在 90% 外点下不崩溃
        self.K_ratio = K_ratio
        self.tol = tol
        self.max_iter = max_iter
        self.n_est = None
        self.d_est = None
        self.inliers = None

    def fit(self, points):
        points = np.asarray(points)
        n_total = points.shape[0]

        # 1. 初始化：直接对所有点进行最小二乘拟合 (No RANSAC)
        # 这是原始 IKOSE 面临高外点时的典型初始状态
        n_curr, d_curr = fit_plane_lsq(points)

        # 2. 计算初始残差并排序
        residuals = compute_residuals(points, n_curr, d_curr)
        abs_res = np.abs(residuals)
        sorted_res = np.sort(abs_res)

        # 3. 确定固定的锚点 r_K (Fixed Anchor)
        # Wang 2012 算法的核心：分子在迭代中是不变的
        K = int(np.ceil(self.K_ratio * n_total))
        K = max(1, min(K, n_total))
        r_K_fixed = sorted_res[K - 1]  # 固定值

        # 初始内点数假设为全部
        n_inlier_prev = n_total
        S_prev = np.inf

        for _ in range(self.max_iter):
            # 4. 计算尺度 S (使用固定的 r_K_fixed 和 动态的 n_inlier)
            kappa = K / n_inlier_prev
            # 限制 kappa 范围防止数值错误
            prob = 0.5 * (1.0 + min(kappa, 1.0))
            phi_inv = norm.ppf(prob)

            if phi_inv < 1e-8: phi_inv = 1e-8
            S_curr = r_K_fixed / phi_inv

            # 5. 更新内点 (硬截断)
            # 原始文献常用 2.5 * S 作为阈值
            inliers_mask = abs_res < 2.5 * S_curr
            n_inlier_curr = np.sum(inliers_mask)

            # 如果内点太少，停止
            if n_inlier_curr < 3:
                break

            # 6. 收敛判断
            if abs(S_curr - S_prev) / S_prev < self.tol:
                # 只有在收敛后，才用最终内点更新一次模型
                # (或者在每次迭代中更新模型，视具体实现而定，这里采用稳健策略：仅更新尺度)
                break

            S_prev = S_curr
            n_inlier_prev = n_inlier_curr

            # 7. (可选) 每次迭代更新模型
            # 虽然原始 IKOSE 是尺度估计器，但为了作为拟合器使用，通常需要更新平面
            # 这样 residuals 才会变，算法才能真正收敛到正确位置
            n_curr, d_curr = fit_plane_lsq(points[inliers_mask])
            residuals = compute_residuals(points, n_curr, d_curr)
            abs_res = np.abs(residuals)
            # 注意：在原始 IKOSE 定义中，r_K_fixed 来源于初始残差，
            # 但为了闭环拟合，通常需要重新计算残差，但 r_K 是否重算有争议。
            # 严格遵循 Wang 2012，r_K 是基于 initial hypothesis 的。
            # 但为了让它能动，这里我们保持 r_K_fixed 不变（基于初始全局分布的锚点），
            # 而让 residuals 随模型变化。

        # 最终拟合
        final_inliers = abs_res < 2.5 * S_prev
        if np.sum(final_inliers) >= 3:
            self.n_est, self.d_est = fit_plane_lsq(points[final_inliers])
        else:
            self.n_est, self.d_est = n_curr, d_curr

        self.inliers = final_inliers
        return self.n_est, self.d_est, self.inliers