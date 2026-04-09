import numpy as np
import os
from scipy.ndimage import gaussian_filter
from osgeo import gdal, osr


class OutlierTerrainSimulator:
    """
    用于生成具有特定外点比例的模拟地形（DEM）和带标签点云的类。
    实现对陨石坑和岩石外点贡献比例的严格控制。
    """

    def __init__(self, size=10, pixel_size=0.3, slope_angle=0.0, threshold=0.1, z=0.0, z_noise_std=0.05, xy_noise_std=0.03):
        self.size = size
        self.pixel_size = pixel_size
        self.slope_angle = slope_angle
        self.threshold = threshold
        self.base_z = z

        # --- PEIV MODIFICATION 1: 新增 Z 和 XY 噪声标准差参数 ---
        self.z_noise_std = z_noise_std
        self.xy_noise_std = xy_noise_std
        # ----------------------------------------------------

        self.base_dem, self.xx, self.yy = self._generate_base_terrain()
        self.rows, self.cols = self.base_dem.shape
        self.total_points = self.rows * self.cols
        self.actual_dem = self.base_dem.copy()  # 实际地形数据

    def _generate_base_terrain(self):
        """生成基础地形（基准面）和坐标网格，并添加 Z 轴随机噪声。"""
        x = np.arange(0, self.size + self.pixel_size, self.pixel_size)
        y = np.arange(0, self.size + self.pixel_size, self.pixel_size)
        xx, yy = np.meshgrid(x, y)

        rows, cols = xx.shape

        slope_rad = np.radians(self.slope_angle)
        base_z = self.base_z
        dem = np.tan(slope_rad) * xx + base_z

        np.random.seed(42)

        # --- PEIV MODIFICATION 2: 简化 Z 轴基础噪声生成 ---
        # 直接使用目标 Z 噪声标准差 (0.05m)
        noise = np.random.normal(0, self.z_noise_std, (rows, cols))
        # 保持高斯平滑以模拟地形的连续性
        noise = gaussian_filter(noise, sigma=0.8)
        dem += noise
        # --------------------------------------------------

        return dem, xx, yy

    def _generate_crater(self, dem, x0, y0, diameter, impact_angle=60):
        """生成单个圆形陨石坑（修改dem）。"""
        R = diameter / 2
        depth = min(1.5, diameter * 0.6)
        rim_height = depth * 0.4

        theta = np.radians(impact_angle)

        # 坐标旋转，使用self.xx, self.yy
        x_rot = (self.xx - x0) * np.cos(theta) + (self.yy - y0) * np.sin(theta)
        y_rot = -(self.xx - x0) * np.sin(theta) + (self.yy - y0) * np.cos(theta)

        mask_scale = 1.1 - (diameter - 0.3) * 0.08 / (5 - 0.3)
        circle_mask = (x_rot ** 2 + y_rot ** 2) <= (R * mask_scale) ** 2

        r = np.sqrt(x_rot ** 2 + y_rot ** 2) / R
        r = np.where(circle_mask, r, 0)

        dem[circle_mask] -= depth * (1 - r[circle_mask] ** 2)

        rim_mask = circle_mask & (r >= 0.6)
        dem[rim_mask] += rim_height * (1 - (r[rim_mask] - 0.6) / 0.5) ** 2

        dem[:] = gaussian_filter(dem, sigma=0.2)
        return circle_mask

    def _generate_rock(self, dem, x0, y0, radius, min_height=0.1):
        """生成单个岩石（修改dem）。"""
        height_factor = np.random.uniform(0.5, 0.8)
        height = max(radius * height_factor, min_height)

        r = np.sqrt((self.xx - x0) ** 2 + (self.yy - y0) ** 2)
        rock_mask = r <= radius * 0.9

        dem[rock_mask] += height * np.exp(-(r[rock_mask] / radius) ** 3)
        dem[:] = gaussian_filter(dem, sigma=0.15)
        return rock_mask

    def _add_noise(self, labels, target_count):
        """在内点上添加随机噪声并更新self.actual_dem。"""
        inlier_indices = np.where(labels == 1)
        available = len(inlier_indices[0])
        if available == 0:
            return 0

        actual_count = min(target_count, available)
        selected_idx = np.random.choice(available, actual_count, replace=False)

        # 噪声幅度降低，避免极端值
        noise = np.random.uniform(0.2, 0.5, size=actual_count)
        self.actual_dem[inlier_indices[0][selected_idx], inlier_indices[1][selected_idx]] += noise

        return actual_count

    def calculate_outliers(self, actual_dem, base_dem):
        """计算点云与基准面的距离，判断外点（1=内点，0=外点）。"""
        distance = np.abs(actual_dem - base_dem)
        # 距离超限为外点 (0)，否则为内点 (1)
        labels = np.where(distance > self.threshold, 0, 1)
        return labels

    # --- 文件 I/O 方法保持不变 ---
    def _save_geotiff(self, data, output_path, origin_x=0, origin_y=0):
        rows, cols = data.shape
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)

        srs = osr.SpatialReference()
        srs.SetWellKnownGeogCS("WGS84")
        out_ds.SetProjection(srs.ExportToWkt())

        geotransform = (origin_x, self.pixel_size, 0, origin_y, 0, -self.pixel_size)
        out_ds.SetGeoTransform(geotransform)

        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(data)
        out_band.FlushCache()
        out_ds = None

    def _save_point_cloud(self, dem, labels, output_path):
        x_flat_nominal = self.xx.flatten()
        y_flat_nominal = self.yy.flatten()
        z_flat = dem.flatten()
        labels_flat = labels.flatten()

        num_points = len(x_flat_nominal)

        # --- PEIV MODIFICATION 3: 为 X 和 Y 添加随机误差 ---
        # 模拟 X 坐标的随机误差 (e_ax)，标准差 0.03m
        dx = np.random.normal(0, self.xy_noise_std, num_points)
        # 模拟 Y 坐标的随机误差 (e_ay)，标准差 0.03m
        dy = np.random.normal(0, self.xy_noise_std, num_points)

        # 生成带有误差的 X 和 Y 观测值
        x_flat_noisy = x_flat_nominal + dx
        y_flat_noisy = y_flat_nominal + dy
        # ----------------------------------------------------

        point_cloud = np.column_stack((x_flat_noisy, y_flat_noisy, z_flat, labels_flat))
        np.savetxt(output_path, point_cloud, fmt="%.4f %.4f %.4f %d")

    # --- 核心生成方法：实现约束逻辑 ---
    def generate_simulation(self, outlier_ratio, dem_dir, pc_dir):
        """
        根据目标外点比例生成地形和点云，并严格控制特征贡献比例。
        """
        # 1. 初始化和目标计算
        self.actual_dem = self.base_dem.copy()
        target_outliers = int(self.total_points * outlier_ratio)

        # 设定严格的比例上限
        MAX_CRATER_OUTLIERS = int(target_outliers * 0.80)
        MAX_ROCK_OUTLIERS = int(target_outliers * 0.20)

        # 确保岩石和陨石坑的上限之和不超过总目标（虽然理论上 80%+20%=100%，但保留灵活性）
        MAX_TOTAL_OUTLIERS_CRATER_ROCK = MAX_CRATER_OUTLIERS + MAX_ROCK_OUTLIERS

        print(f"\n--- 目标比例: {outlier_ratio:.2f} (点数: {target_outliers}) ---")
        print(f"  > 陨石坑上限: {MAX_CRATER_OUTLIERS} | 岩石上限: {MAX_ROCK_OUTLIERS}")

        current_outliers = 0

        # --------------------------
        # 1. 步骤一：添加陨石坑 (Max 80%)
        # --------------------------
        attempts = 0
        max_attempts = 30000

        while current_outliers < MAX_CRATER_OUTLIERS and attempts < max_attempts:
            attempts += 1

            # 1a. 保存当前地形状态，用于回滚
            dem_before = self.actual_dem.copy()

            # 1b. 计算陨石坑参数 (沿用原逻辑的动态尺寸)
            ratio_factor = min(target_outliers / 1000, 0.9)
            min_dia = 0.5 + (ratio_factor - 0.1) * (2.0 - 0.5) / 0.8
            max_dia = 1.0 + (ratio_factor - 0.1) * (10.0 - 1.0) / 0.8
            diameter = np.random.uniform(min_dia, max_dia)
            margin = diameter * 0.05
            x0 = np.random.uniform(margin, self.size - margin)
            y0 = np.random.uniform(margin, self.size - margin)

            # 1c. 生成单个陨石坑 (修改 self.actual_dem)
            self._generate_crater(self.actual_dem, x0, y0, diameter)

            # 1d. 检查新的总外点数
            new_labels = self.calculate_outliers(self.actual_dem, self.base_dem)
            new_outliers = np.sum(new_labels == 0)

            if new_outliers <= MAX_CRATER_OUTLIERS:
                current_outliers = new_outliers
            else:
                # 违反约束，回滚地形，并停止添加陨石坑
                self.actual_dem = dem_before
                break

        actual_crater_outliers = current_outliers
        print(f"  > 陨石坑后：实际外点={current_outliers} (贡献率: {actual_crater_outliers / target_outliers * 100:.1f}%)")

        # --------------------------
        # 2. 步骤二：添加岩石 (Max 20%)
        # --------------------------

        # 岩石总目标：不能超过陨石坑已贡献 + 岩石最大允许贡献
        max_total_outliers_after_rocks = actual_crater_outliers + MAX_ROCK_OUTLIERS

        attempts = 0
        max_attempts = 20000

        while current_outliers < max_total_outliers_after_rocks and attempts < max_attempts:
            attempts += 1

            dem_before = self.actual_dem.copy()

            # 2b. 计算岩石参数
            ratio_factor = min(target_outliers / 1000, 0.9)
            min_rad = 0.1 + (ratio_factor - 0.1) * (0.3 - 0.1) / 0.8
            max_rad = 0.5 + (ratio_factor - 0.1) * (0.8 - 0.5) / 0.8
            radius = np.random.uniform(min_rad, max_rad)
            margin = radius * 0.3
            x0 = np.random.uniform(margin, self.size - margin)
            y0 = np.random.uniform(margin, self.size - margin)

            # 2c. 生成单个岩石
            self._generate_rock(self.actual_dem, x0, y0, radius)

            # 2d. 检查新的总外点数
            new_labels = self.calculate_outliers(self.actual_dem, self.base_dem)
            new_outliers = np.sum(new_labels == 0)

            if new_outliers <= max_total_outliers_after_rocks:
                current_outliers = new_outliers
            else:
                # 违反约束，回滚地形，并停止添加岩石
                self.actual_dem = dem_before
                break

        actual_rock_outliers = current_outliers - actual_crater_outliers
        print(f"  > 岩石后：实际外点={current_outliers} (岩石贡献: {actual_rock_outliers / target_outliers * 100:.1f}%)")

        # --------------------------
        # 3. 步骤三：添加噪声 (补充剩余需求)
        # --------------------------

        remaining_needed = target_outliers - current_outliers

        if remaining_needed > 0:
            # 基于当前地形计算内点，确保噪声只影响内点
            temp_labels = self.calculate_outliers(self.actual_dem, self.base_dem)
            actual_noise_points = self._add_noise(temp_labels, remaining_needed)

            # 最终计算总外点
            final_labels = self.calculate_outliers(self.actual_dem, self.base_dem)
            current_outliers = np.sum(final_labels == 0)
        else:
            final_labels = self.calculate_outliers(self.actual_dem, self.base_dem)
            actual_noise_points = 0

        actual_noise_outliers = current_outliers - actual_crater_outliers - actual_rock_outliers

        print(f"  > 噪声后：实际外点={current_outliers} (噪声贡献: {actual_noise_outliers / target_outliers * 100:.1f}%)")

        # --------------------------
        # 4. 统计与保存结果
        # --------------------------
        actual_ratio = current_outliers / self.total_points * 100
        pct = int(round(outlier_ratio * 100))

        dem_path = os.path.join(dem_dir, f"dem_{pct}pct.tif")
        self._save_geotiff(self.actual_dem, dem_path)

        pc_path = os.path.join(pc_dir, f"point_cloud_{pct}pct.txt")
        self._save_point_cloud(self.actual_dem, final_labels, pc_path)

        print(
            f"  最终结果：实际外点={current_outliers}（{actual_ratio:.1f}%）"
        )


def main():
    dem_dir = "dem_files_output_peiv15"
    pc_dir = "point_clouds_output_peiv15"
    os.makedirs(dem_dir, exist_ok=True)
    os.makedirs(pc_dir, exist_ok=True)

    # --- PEIV MODIFICATION 4: 设置 Z 和 XY 噪声标准差 ---
    # Z 噪声标准差：0.05m
    # X/Y 噪声标准差：0.03m
    simulator = OutlierTerrainSimulator(
        slope_angle=35,
        z=30,
        threshold=0.05,
        z_noise_std=0.08,
        xy_noise_std=0.05
    )
    # ----------------------------------------------------

    # 生成10%~90%外点（每10%一组，用于测试约束效果）
    for ratio in np.arange(0.10, 0.91, 0.01):
        simulator.generate_simulation(ratio, dem_dir, pc_dir)

    print(f"\n所有数据生成完成：\nDEM文件保存至 {os.path.abspath(dem_dir)}\n点云文件保存至 {os.path.abspath(pc_dir)}")


if __name__ == "__main__":
    main()