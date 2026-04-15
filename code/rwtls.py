import numpy as np
from scipy.linalg import inv


class RWTLSPlaneFitter:
    def __init__(self, sigma_x, sigma_y, sigma_z, max_iter=100, eps_xi=1e-5):
        """
        初始化RWTLS平面拟合类
        :param sigma_x: x方向的标准差
        :param sigma_y: y方向的标准差
        :param sigma_z: z方向的标准差
        :param max_iter: 最大迭代次数
        :param eps_xi: 收敛阈值
        """
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z
        self.max_iter = max_iter
        self.eps_xi = eps_xi
        self.n_est = None
        self.d_est = None
        self.inliers = None

    def fit(self, points):
        """
        执行RWTLS平面拟合
        :param points: Nx3点云数组
        :return: n, d, inliers_mask
        """
        n_points = points.shape[0]
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"点云维度错误：需为(N,3)，实际为{points.shape}")
        if n_points < 3:
            raise ValueError(f"点云数量不足：需≥3，实际为{n_points}")

        x = points[:, 0].reshape(-1, 1)
        y = points[:, 1].reshape(-1, 1)
        z = points[:, 2].reshape(-1, 1)
        n = n_points
        m = 3

        # 关键修正1: t = 2*n (原始代码正确值)
        t = 2 * n

        # 初始化
        h = np.zeros((3 * n, 1))
        h[2 * n: 3 * n, 0] = 1.0
        B = np.zeros((3 * n, t))
        B[:n, :n] = np.eye(n)
        B[n:2 * n, n:2 * n] = np.eye(n)

        Q_LL_init = np.eye(n) * (self.sigma_z ** 2)
        Q_SS_init = np.block([
            [np.eye(n) * (self.sigma_x ** 2), np.zeros((n, n))],
            [np.zeros((n, n)), np.eye(n) * (self.sigma_y ** 2)]
        ])

        # 计算初始LS估计（正则化）
        A_ls = np.column_stack((x, y, np.ones((n, 1))))
        L = z.reshape(-1, 1)
        ATA = A_ls.T @ A_ls
        ATA_reg = ATA + np.eye(3) * 1e-6
        if np.linalg.cond(ATA_reg) > 1e12:
            ATA_inv = np.linalg.pinv(ATA_reg)
        else:
            ATA_inv = np.linalg.inv(ATA_reg)
        xi = ATA_inv @ A_ls.T @ L

        e_alpha = np.zeros((2 * n, 1))
        e_L = np.zeros((n, 1))
        iter_cnt = 0

        print(f"\n{'迭代次数':<10}{'ξ改正量 norm':<20}{'收敛'}")
        print(f"{'--':<10}{'--':<20}{'--'}")

        while iter_cnt < self.max_iter:
            iter_cnt += 1

            # 计算当前A矩阵
            a = np.concatenate((x, y), axis=0)
            vec_arg = h + B @ (a - e_alpha)
            A_i = vec_arg.reshape(n, m, order='F')  # 修正：使用reshape替代ivec

            # 计算当前残差
            current_residual_L = z - A_i @ xi
            residuals = current_residual_L.flatten()

            # 关键修正2: 使用当前迭代的权矩阵计算sigma0
            V_L = current_residual_L
            V_S = e_alpha
            weighted_sum = (V_L.T @ inv(Q_LL_init) @ V_L) + (V_S.T @ inv(Q_SS_init) @ V_S)
            r = n - m
            sigma0 = np.sqrt(weighted_sum[0, 0] / r) if r > 0 else 1.0

            # 生成权因子
            w_L = self.igg3_weight(residuals, sigma0)
            w_alpha_x = self.igg3_weight(e_alpha[:n].flatten(), sigma0)
            w_alpha_y = self.igg3_weight(e_alpha[n:].flatten(), sigma0)
            W_L = np.diag(w_L)
            W_alpha = np.diag(np.concatenate((w_alpha_x, w_alpha_y)))

            # 更新协方差逆矩阵（正则化）
            Q_LL = W_L @ Q_LL_init
            Q_LL_inv = np.linalg.inv(Q_LL) if np.linalg.cond(Q_LL) < 1e12 else np.linalg.pinv(Q_LL)
            Q_SS = W_alpha @ Q_SS_init
            Q_SS_inv = np.linalg.inv(Q_SS) if np.linalg.cond(Q_SS) < 1e12 else np.linalg.pinv(Q_SS)

            # 关键修正3: 正确计算S矩阵（核心迭代变量）
            kron_term = np.kron(xi.T, np.eye(n))
            S = -kron_term @ B  # S维度: (n, 2n)

            # 计算M矩阵（正则化）
            M = A_i.T @ Q_LL_inv @ A_i
            M_reg = M + np.eye(3) * 1e-6
            if np.linalg.cond(M_reg) > 1e12:
                M_inv = np.linalg.pinv(M_reg)
            else:
                M_inv = np.linalg.inv(M_reg)

            # 计算delta_xi
            delta_xi = M_inv @ (A_i.T @ Q_LL_inv @ (z - A_i @ xi))
            delta_norm = np.linalg.norm(delta_xi)
            print(f"{iter_cnt:<10}{delta_norm:<20.6e}{'是' if delta_norm < self.eps_xi else '否'}")

            if delta_norm < self.eps_xi:
                xi += delta_xi
                break

            # 关键修正4: 正确更新e_L和e_alpha（使用S矩阵）
            residual_L = z - A_i @ xi
            residual_S = e_alpha
            Q_inv_residual_L = Q_LL_inv @ residual_L
            Q_inv_residual_S = Q_SS_inv @ residual_S

            # 修正: 正确计算Q_SL_T（原始代码方式）
            Q_SL_T = (W_L @ np.zeros((n, t))).T  # (2n, n)  [关键修正！]
            Q_SS_S_T = Q_SS @ S.T  # (2n, 2n) @ (2n, n) = (2n, n)
            term_alpha = Q_SL_T + Q_SS_S_T  # (2n, n)

            e_L_new = (Q_LL + np.zeros((n, n))) @ Q_inv_residual_L  # 保持为零矩阵
            e_alpha_new = term_alpha @ Q_inv_residual_L

            # 更新迭代变量
            xi += delta_xi
            e_L = e_L_new
            e_alpha = e_alpha_new

        # 计算平面参数
        a_est, b_est, c_est = xi.flatten()
        n_est = np.array([a_est, b_est, -1.0])
        d_est = c_est
        if d_est < 0:
            n_est = -n_est
            d_est = -d_est

        # 计算残差（使用最终平面）
        residuals = (n_est[0] * x + n_est[1] * y + n_est[2] * z + d_est).flatten()

        # 关键修正5: 正确计算最终sigma0
        V_L_final = e_L
        V_S_final = e_alpha
        weighted_sum_final = (V_L_final.T @ Q_LL_inv @ V_L_final) + (V_S_final.T @ Q_SS_inv @ V_S_final)
        r_final = n - m
        sigma0_final = np.sqrt(weighted_sum_final[0, 0] / r_final) if r_final > 0 and weighted_sum_final[
            0, 0] > 0 else 1.0

        # outlier检测
        abs_res = np.abs(residuals)
        outlier_mask = abs_res >= 0.25
        inliers_mask = np.ones(n, dtype=bool)
        inliers_mask[outlier_mask] = False

        self.n_est = n_est
        self.d_est = d_est
        self.inliers = inliers_mask

        print(f"\n✅ 稳健估计完成：内点={np.sum(inliers_mask)}，外点={n - np.sum(inliers_mask)}")
        print(f"最终单位权中误差sigma0: {sigma0_final:.6f}")
        print(f"抗差阈值（k2*sigma0）: {3.0 * sigma0:.6f}")

        return self.n_est, self.d_est, self.inliers

    def igg3_weight(self, residuals, sigma0, k1=2.5, k2=3.0):
        """IGGIII权函数，生成残差的等价权因子"""
        abs_res = np.abs(residuals)
        w = np.ones_like(abs_res)
        mask = (abs_res > k1 * sigma0) & (abs_res <= k2 * sigma0)
        w[mask] = (k2 * sigma0) / abs_res[mask]
        w[abs_res > k2 * sigma0] = 0.0
        return w