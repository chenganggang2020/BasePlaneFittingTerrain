import numpy as np
from scipy.stats import norm


def igg3_weight(residuals, sigma0, k1=2.0, k2=4.0):
    """
    IGG3 稳健权函数
    residuals: 应该是标准化后的残差，或者是与 sigma0 同量纲的残差
    sigma0: 必须是标量 (Scalar)，或者与 residuals 形状完全一致的数组
    """
    abs_res = np.abs(residuals)
    w = np.ones_like(abs_res)

    # 降权域
    mask_down = (abs_res > k1 * sigma0) & (abs_res <= k2 * sigma0)

    # 修复点：确保 sigma0 是标量，或者能被正确广播
    if np.any(mask_down):
        # 如果 sigma0 是标量，这里直接除没问题
        # 如果 sigma0 是数组，需要切片 sigma0[mask_down]
        if np.isscalar(sigma0):
            denom = sigma0
        else:
            denom = sigma0[mask_down]

        u = abs_res[mask_down] / denom
        w[mask_down] = (k1 / u) * ((k2 - u) / (k2 - k1)) ** 2

    # 淘汰域
    w[abs_res > k2 * sigma0] = 0.0
    return w


def robust_sigma0(residuals, n, p):
    """
    使用 MAD (Median Absolute Deviation) 估计单位权中误差
    residuals: 建议传入标准化残差 (v / sigma_total)
    """
    median_abs = np.median(np.abs(residuals))
    # 0.6745 是正态分布的分位数 correction factor (1/Phi^-1(0.75))
    sigma = median_abs / 0.6745
    return sigma if sigma > 1e-12 else 1.0


class SequentialRobustTLSPlaneFitter:
    def __init__(self, sigma_x, sigma_y, sigma_z, k1=2.5, k2=4.5, max_iter=100, eps_xi=1e-8):
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z
        self.k1 = k1
        self.k2 = k2
        self.max_iter = max_iter
        self.eps_xi = eps_xi
        self.n_est = None
        self.d_est = None
        self.inliers = None

    def fit(self, points):
        points = np.asarray(points)
        n = points.shape[0]

        # 原始观测值 L = z, a = [x, y]
        x_obs = points[:, 0].reshape(-1, 1)
        y_obs = points[:, 1].reshape(-1, 1)
        z_obs = points[:, 2].reshape(-1, 1)

        # 协因数阵分量
        Q_LL = np.full(n, self.sigma_z ** 2)
        Q_xx = np.full(n, self.sigma_x ** 2)
        Q_yy = np.full(n, self.sigma_y ** 2)

        # --- 1. 初始估计 (LS) ---
        A_init = np.column_stack((x_obs, y_obs, np.ones((n, 1))))
        xi = np.linalg.lstsq(A_init, z_obs, rcond=None)[0]

        # 初始化误差向量 e_a
        e_x = np.zeros((n, 1))
        e_y = np.zeros((n, 1))

        sigma0 = 1.0

        print(f"{'Iter':<5} {'Delta_Xi':<15} {'Sigma0':<10}")
        print("-" * 35)

        for i in range(self.max_iter):
            # 获取当前参数
            a_param = xi[0, 0]
            b_param = xi[1, 0]

            # --- Step 1: 构造/更新设计矩阵 A(i) ---
            current_x = x_obs - e_x
            current_y = y_obs - e_y
            A_curr = np.column_stack((current_x, current_y, np.ones((n, 1))))

            # --- Step 2: 计算闭合差 (Misclosure) ---
            l_misclosure = z_obs - A_curr @ xi

            # --- Step 3: 构建协因数阵与定权 (关键修复步骤) ---

            # 1. 计算每个点的总方差 (Variance Total)
            # 这代表了点位误差在当前法向量方向上的投影总和
            variance_total = Q_LL + (a_param ** 2 * Q_xx) + (b_param ** 2 * Q_yy)
            sigma_total_vec = np.sqrt(variance_total)  # 这是一个数组 (n,)

            # 2. 计算标准化残差 (Standardized Residuals)
            # 将残差除以其理论标准差，使其量纲归一化
            std_residuals = l_misclosure.flatten() / sigma_total_vec

            # 3. 估计单位权中误差 (Sigma0)
            # 此时 sigma0 是一个标量 (Scalar)
            sigma0 = robust_sigma0(std_residuals, n, 3)

            # 4. 计算 IGG3 权重
            # 关键修改：直接传入标准化残差和标量 sigma0
            # 这样 igg3_weight 内部比较的是: |v_std| vs k * sigma0
            w_robust = igg3_weight(std_residuals, sigma0, self.k1, self.k2)

            # --- Step 4: 构建总权阵 P ---
            # P = w_robust * (1 / Variance_total)
            P_diag = w_robust / variance_total

            # --- Step 5: 计算参数修正量 ---
            P_mat = P_diag.reshape(-1, 1)
            ATA = A_curr.T @ (A_curr * P_mat)
            ATl = A_curr.T @ (l_misclosure * P_mat)

            try:
                # 添加微小正则化，防止病态
                delta_xi = np.linalg.solve(ATA + np.eye(3) * 1e-12, ATl)
            except np.linalg.LinAlgError:
                delta_xi = np.linalg.lstsq(ATA, ATl, rcond=None)[0]

            xi_new = xi + delta_xi

            if np.linalg.norm(delta_xi) < self.eps_xi:
                xi = xi_new
                print(f"{i:<5} {np.linalg.norm(delta_xi):<15.8f} {sigma0:<10.4f} Converged")
                break

            print(f"{i:<5} {np.linalg.norm(delta_xi):<15.8f} {sigma0:<10.4f}")

            # --- Step 6: 更新误差向量 e_a (公式 3.69 & 3.70) ---
            # 使用新参数计算总残差
            r_total = z_obs - A_curr @ xi_new

            # 计算拉格朗日乘数 K
            # K = Q1^-1 * r_total (含稳健权重)
            K = P_diag.reshape(-1, 1) * r_total

            # 反向传播误差
            a_new, b_new = xi_new[0, 0], xi_new[1, 0]

            e_x = Q_xx.reshape(-1, 1) * a_new * K
            e_y = Q_yy.reshape(-1, 1) * b_new * K

            xi = xi_new

        # --- 结果整理 ---
        self.n_est = np.array([xi[0, 0], xi[1, 0], -1.0])
        self.d_est = xi[2, 0]

        # norm_val = np.linalg.norm(self.n_est)
        # self.n_est = self.n_est / norm_val
        # self.d_est = self.d_est / norm_val

        if self.d_est < 0:
            self.n_est = -self.n_est
            self.d_est = -self.d_est

        # 计算最终内点
        # 使用正交距离判断
        denom = np.sqrt(xi[0, 0] ** 2 + xi[1, 0] ** 2 + 1)
        ortho_dist = np.abs(x_obs * xi[0, 0] + y_obs * xi[1, 0] + xi[2, 0] - z_obs) / denom

        # 3倍中误差作为阈值 (这里的 sigma0 已经是单位权中误差，需要乘上位置误差比例?)
        # 简单起见，直接用 sigma0 * 平均点位精度估计
        mean_pos_sigma = np.mean(sigma_total_vec)
        self.inliers = (ortho_dist.flatten() < 3.0 * sigma0 * mean_pos_sigma)

        return self.n_est, self.d_est, self.inliers