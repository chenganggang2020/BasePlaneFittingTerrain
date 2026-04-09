import numpy as np

class WTLSPlaneFitter:
    def __init__(self, sigma_x, sigma_y, sigma_z, max_iter=10, eps_xi=1e-5):
        """
        初始化WTLS平面拟合类
        :param sigma_x: x方向的中误差
        :param sigma_y: y方向的中误差
        :param sigma_z: z方向的中误差
        :param max_iter: 最大迭代次数
        :param eps_xi: 参数改正量收敛阈值
        """
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z
        self.max_iter = max_iter
        self.eps_xi = eps_xi
        self.n_est = None
        self.d_est = None
        self.inliers = None

    def ivec(self, vec, shape):
        """vec的逆操作：将向量按列重塑为矩阵（对应文献中ivec）"""
        return vec.reshape(shape, order='F')

    def fit(self, points):
        """
        执行WTLS平面拟合
        :param points: Nx3点云数组
        :return: n, d, inliers_mask
        """
        # 维度和点数校验
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"点云维度错误：需为(N,3)，实际为{points.shape}")
        n = points.shape[0]
        if n < 3:
            raise ValueError(f"点云数量不足：需≥3，实际为{n}")

        x = points[:, 0].reshape(-1, 1)
        y = points[:, 1].reshape(-1, 1)
        z = points[:, 2].reshape(-1, 1)
        t = 2 * n  # 变量误差总个数（x/y各n个）
        m = 3      # 平面参数个数（ξ=[a,b,c]^T）

        # 构造h（固定向量）和B（误差映射矩阵）
        h = np.zeros((3 * n, 1))
        h[2 * n: 3 * n, 0] = 1.0
        B = np.zeros((3 * n, t))
        B[:n, :n] = np.eye(n)
        B[n:2 * n, n:2 * n] = np.eye(n)

        # 协方差矩阵Q（权阵P = Q^-1）
        Q_LL = np.eye(n) * (self.sigma_z ** 2)    # z观测误差的协方差
        Q_LS = np.zeros((n, t))                   # 观测与变量误差的协方差（假设无关）
        Q_SL = Q_LS.T
        Q_SS = np.block([
            [np.eye(n) * (self.sigma_x ** 2), np.zeros((n, n))],
            [np.zeros((n, n)), np.eye(n) * (self.sigma_y ** 2)]
        ])  # x/y变量误差的协方差
        Q = np.block([[Q_LL, Q_LS], [Q_SL, Q_SS]])
        Q_inv = np.linalg.inv(Q)  # Q的逆，用于加权计算
        Q_LL_inv = np.linalg.inv(Q_LL)

        # 初始值（LS估计，忽略变量误差）
        A_ls = np.column_stack((x, y, np.ones((n, 1))))
        xi = np.linalg.lstsq(A_ls, z, rcond=None)[0].reshape(m, 1)
        e_alpha = np.zeros((2 * n, 1))
        e_L = np.zeros((n, 1))

        # 迭代参数
        eps_xi = self.eps_xi
        max_iter = self.max_iter
        iter_cnt = 0

        print(f"\n{'迭代次数':<10}{'ξ改正量 norm':<20}{'收敛'}")
        print(f"{'--':<10}{'--':<20}{'--'}")

        while iter_cnt < max_iter:
            iter_cnt += 1

            # 计算A_(i)：ivec(h + B(a - e_alpha))
            a = np.concatenate((x, y))
            vec_arg = h + B @ (a - e_alpha.reshape(-1, 1))
            A_i = self.ivec(vec_arg, (n, m))

            # 构造G和S（公式3.67）
            kron_term = np.kron(xi.T, np.eye(n))
            S = -kron_term @ B
            G = np.hstack([np.eye(n), S])

            # 构造L向量
            L_vec = np.concatenate((z - e_L, e_alpha.reshape(-1, 1)))

            # 计算A_(i)ξ_(i)
            A_i_xi = A_i @ xi

            # 求解参数改正量δξ（公式3.68）
            delta_xi = np.linalg.inv(A_i.T @ Q_LL_inv @ A_i) @ A_i.T @ Q_LL_inv @ (z - A_i_xi.reshape(-1, 1))

            # 收敛判断
            delta_xi_norm = np.linalg.norm(delta_xi)
            converged = delta_xi_norm < eps_xi
            print(f"{iter_cnt:<10}{delta_xi_norm:<20.6e}{'是' if converged else '否'}")

            if converged:
                xi = xi + delta_xi
                break

            # 求解观测误差e_L和变量误差e_alpha的新估计（公式3.69、3.70）
            residual = L_vec - A_i_xi.reshape(-1, 1) - A_i @ delta_xi
            Q_inv_residual = Q_inv @ residual

            # 更新e_L(i+1)
            Q_LS_S_T = Q_LS @ S.T
            term_LL = Q_LL + Q_LS_S_T
            e_L_new = term_LL @ Q_inv_residual[:n]

            # 更新e_alpha(i+1)
            Q_SL_T = Q_SL.T
            Q_SS_S_T = Q_SS @ S.T
            term_alpha = Q_SL_T + Q_SS_S_T
            e_alpha_new = term_alpha @ Q_inv_residual[:n]

            # 迭代更新
            xi = xi + delta_xi
            e_L = e_L_new.reshape(-1, 1)
            e_alpha = e_alpha_new.reshape(-1)

        # 计算平面参数
        a_est, b_est, c_est = xi.flatten()
        n_est = np.array([a_est, b_est, -1.0])

        d_est = c_est

        # 计算残差（点到平面的带符号距离）
        residuals = (n_est[0] * x + n_est[1] * y + n_est[2] * z + d_est).flatten()

        # 计算验证单位权中误差
        V = np.concatenate((z - e_L, e_alpha.reshape(-1, 1)))
        r = n - 3
        sigma0 = np.sqrt((V.T @ Q_inv @ V) / r) if r > 0 else 1.0

        # 内点判定（基于残差绝对值）
        abs_res = np.abs(residuals)
        outlier_mask = abs_res >= 0.25
        inliers_mask = np.ones(n, dtype=bool)
        if outlier_mask.ndim == 2:
            outlier_mask = np.ravel(outlier_mask)
        inliers_mask[outlier_mask] = False

        self.n_est = n_est
        self.d_est = d_est

        self.inliers = inliers_mask

        print(f"\n✅ 估计完成：内点={np.sum(inliers_mask)}，外点={n - np.sum(inliers_mask)}")
        print(f"最终单位权中误差sigma0: {sigma0:.6f}")

        return self.n_est, self.d_est, self.inliers
