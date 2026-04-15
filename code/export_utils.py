import numpy as np
import os
from scipy.io import savemat


def export_to_cloudcompare(points, inliers, n, d, method_name, root_dir="results"):
    """
    导出点云和拟合平面到CloudCompare兼容格式（保存到results/cloudcompare子文件夹）
    - XYZ格式点云文件（带RGB颜色标记，内点绿色，外点红色）
    - OFF格式平面网格文件
    """
    # 创建CloudCompare专用子文件夹（results/cloudcompare）
    output_dir = os.path.join(root_dir, "cloudcompare")
    os.makedirs(output_dir, exist_ok=True)

    # 1. 导出带颜色的点云（XYZ格式）
    # 数据结构：x y z r g b（颜色值0-255，内点绿色，外点红色）
    point_data = np.hstack([points, np.zeros((points.shape[0], 3), dtype=np.uint8)])
    point_data[inliers, 3:6] = [0, 255, 0]  # 内点：绿色(R=0, G=255, B=0)
    point_data[~inliers, 3:6] = [255, 0, 0]  # 外点：红色(R=255, G=0, B=0)

    xyz_path = os.path.join(output_dir, f"{method_name}_points.xyz")
    np.savetxt(
        xyz_path,
        point_data,
        fmt="%.4f %.4f %.4f %d %d %d",  # 格式：x(4位小数) y z r g b(整数)
        header="X Y Z R G B"  # 表头说明字段含义
    )
    print(f"CloudCompare点云导出成功：{xyz_path}")

    # 2. 导出拟合平面（OFF格式）
    # 计算平面网格范围（基于点云坐标范围）
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    x_grid = np.linspace(x_min, x_max, 50)  # 50x50网格足够展示平面
    y_grid = np.linspace(y_min, y_max, 50)
    X, Y = np.meshgrid(x_grid, y_grid)

    # 根据平面方程计算Z值（ax + by + cz + d = 0 → z = (-ax - by - d)/c）
    if np.abs(n[2]) < 1e-8:
        # 处理法向量z分量为0的特殊情况（避免除零）
        Z = np.zeros_like(X)
    else:
        Z = (-n[0] * X - n[1] * Y - d) / n[2]

    # 生成平面顶点坐标
    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    num_vertices = vertices.shape[0]  # 50x50=2500个顶点

    # 生成三角形面片索引（将网格分割为三角形）
    faces = []
    for i in range(len(x_grid) - 1):
        for j in range(len(y_grid) - 1):
            # 每个网格单元的4个顶点索引
            idx1 = i * len(y_grid) + j
            idx2 = idx1 + 1
            idx3 = idx1 + len(y_grid)
            idx4 = idx3 + 1
            # 分割为两个三角形（OFF格式要求每个面片首数字为顶点数）
            faces.append([3, idx1, idx2, idx3])  # 第一个三角形
            faces.append([3, idx2, idx4, idx3])  # 第二个三角形
    num_faces = len(faces)

    # 保存OFF文件
    off_path = os.path.join(output_dir, f"{method_name}_plane.off")
    with open(off_path, "w") as f:
        f.write("OFF\n")  # OFF格式标识（必须首行）
        f.write(f"{num_vertices} {num_faces} 0\n")  # 顶点数 面片数 边数（边数可设为0）
        # 写入顶点坐标
        for v in vertices:
            f.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        # 写入面片索引
        for face in faces:
            f.write(f"{' '.join(map(str, face))}\n")

    print(f"CloudCompare平面导出成功：{off_path}")


def export_to_matlab(points, inliers, n, d, method_name, root_dir="results"):
    """
    导出点云和平面参数到MATLAB兼容的.mat文件（保存到results/matlab子文件夹）
    包含点云坐标、内点索引、平面法向量、平面方程参数
    """
    # 创建MATLAB专用子文件夹（results/matlab）
    output_dir = os.path.join(root_dir, "matlab")
    os.makedirs(output_dir, exist_ok=True)

    # 整理需要保存的数据（与MATLAB变量名对应）
    mat_data = {
        'points': points,  # 原始点云坐标 (N×3数组)
        'inliers': inliers.astype(bool),  # 内点索引 (N×1布尔数组)
        'plane_normal': n,  # 平面法向量 (3×1数组)
        'plane_d': d,  # 平面截距（标量）
        'plane_equation': np.array([n[0], n[1], n[2], d])  # 平面方程 ax+by+cz+d=0
    }

    # 保存MAT文件
    mat_path = os.path.join(output_dir, f"{method_name}_results.mat")
    savemat(mat_path, mat_data)  # scipy的savemat函数确保MATLAB兼容

    print(f"MATLAB文件导出成功：{mat_path}")
