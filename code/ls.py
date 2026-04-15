import numpy as np
from scipy.linalg import inv

class LSPlaneFitter:
    def __init__(self):
        """
        初始化LS平面拟合器
        """
        self.n = None
        self.d = None
        self.inliers = None

    def fit(self, points):
        """
        执行最小二乘平面拟合
        :param points: Nx3点云数组
        :return: n（法向量），d（截距），inliers（全为True的布尔数组）
        """
        N = points.shape[0]
        if N < 3:
            raise ValueError("点云数量需≥3")

        # 构建设计矩阵A和观测向量L
        A = np.hstack([points[:, :2], np.ones((N, 1))])  # A = [x y 1]
        L = points[:, 2].reshape(-1, 1)  # z

        # 最小二乘参数估计：X = (A^T A)^-1 A^T L
        ATA = A.T @ A
        # 避免矩阵奇异，加入微小扰动
        if np.linalg.cond(ATA) > 1e10:
            ATA += np.eye(3) * 1e-8
        ATA_inv = inv(ATA)
        X = ATA_inv @ A.T @ L  # [a, b, c]^T

        a, b, c = X[0, 0], X[1, 0], X[2, 0]

        # 计算初始法向量（单位向量）
        n = np.array([a, b, -1.0])

        # 计算截距d（确保为正）
        d = -np.mean(np.dot(points, n))
        if d < 0:
            n = -n
            d = -d

        # 所有点视为内点（无抗差）
        inliers = np.ones(N, dtype=bool)

        # 保存结果
        self.n = n
        self.d = d
        self.inliers = inliers

        return self.n, self.d, self.inliers
