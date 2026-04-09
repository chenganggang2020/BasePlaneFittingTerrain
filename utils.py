import numpy as np
from scipy.linalg import lstsq
import os

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None


def plane_from_3points(points):
    """由3个非共线点计算平面参数（n, d），平面方程：n·x + d = 0"""
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    n = np.cross(v1, v2)
    d = -np.dot(n, points[0])
    return n, d


def compute_residuals(points, n, d):
    # 计算点到平面的残差（距离）
    numerator = np.abs(n[0] * points[:,0] + n[1] * points[:,1] + n[2] * points[:,2] + d)
    denominator = np.linalg.norm(n)
    return numerator / denominator  # 仅返回数组，无任何布尔判断


def fit_plane_lsq(points):
    """最小二乘平面拟合（用内点集优化平面参数）"""
    A = np.hstack([points[:, :2], np.ones((points.shape[0], 1))])
    b = -points[:, 2]
    x, _, _, _ = lstsq(A, b, cond=1e-10)
    nx, ny, d = x[0], x[1], x[2]
    nz = 1.0
    n = np.array([nx, ny, nz])
    d = -np.mean(np.dot(points, n))
    return n, d


# -------------------------- 原有辅助函数（保持不变，根据实际代码补充） --------------------------
def load_point_cloud(file_path):
    """加载点云文件（自动识别逗号/空格分隔，支持带标签或不带标签格式）"""
    try:
        # 1. 先尝试读取逗号分隔的文件（优先处理用户之前的格式）
        data = np.loadtxt(file_path, delimiter=',')
        delimiter_used = "逗号"  # 记录实际使用的分隔符，用于错误提示
    except ValueError as e:
        # 若逗号分隔读取失败（如文件是空格分隔），尝试默认的空格/空白字符分隔
        try:
            data = np.loadtxt(file_path)  # 默认分隔符：任意空白字符（空格、制表符等）
            delimiter_used = "空格/空白字符"
        except Exception as e2:
            # 两种分隔符都失败时，抛出包含详细信息的异常
            raise type(e2)(
                f"读取点云文件失败：尝试了逗号分隔和空格分隔均失败。\n"
                f"错误详情：{str(e2)}\n"
                f"请检查文件格式：确保是纯数值（x y z [label]），且分隔符为逗号或空格。"
            ) from e2
    except Exception as e:
        # 捕获其他非格式错误（如文件不存在、权限问题）
        raise type(e)(
            f"读取点云文件失败：{str(e)}\n"
            f"请检查文件路径是否正确，或是否有读取权限。"
        ) from e

    # 提取点云和标签（逻辑不变）
    points = data[:, :3]  # 前3列是x、y、z坐标
    # 检查是否有第4列（标签列：1=外点，0=内点，无标签则默认全为内点）
    if data.shape[1] > 3:
        labels = data[:, 3].astype(int)
    else:
        labels = np.ones(len(points), dtype=int)  # 无标签时默认所有点为内点

    print(f"成功读取点云文件：共{len(points)}个点，使用{delimiter_used}分隔")
    return points, labels


def read_point_cloud(file_path):
    """Read point cloud file"""
    try:
        if pd is None:
            points = np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=[0, 1, 2])
            points = np.atleast_2d(points)
        else:
            df = pd.read_csv(
                file_path, sep=",", header=0, usecols=[0, 1, 2],
                skip_blank_lines=True, engine='python'
            )
            points = df.values
        points = points[~np.isnan(points).any(axis=1)]
        if points.shape[0] < 3:
            raise ValueError(f"Insufficient valid points: {points.shape[0]} found (need at least 3)")
        print(f"Successfully read point cloud file: {file_path}, Number of points: {points.shape[0]}")
        return points
    except Exception as e:
        raise RuntimeError(f"Failed to read point cloud: {str(e)}")


def downsample_data(points, residuals=None, step=10, min_points=10000, min_density=50):
    """
    智能降采样：根据点云数量和密度自动判断是否需要降采样
    :param points: 原始点云数据 (N, 3)
    :param residuals: 对应的残差数据 (N,)，可选
    :param step: 降采样步长（越大保留点数越少）
    :param min_points: 触发降采样的最小点数阈值（低于此值不降采样）
    :param min_density: 触发降采样的最小密度阈值（点/平方米，低于此值不降采样）
    :return: 降采样后的点云和残差（若提供），或原始数据
    """
    n_points = points.shape[0]

    # 1. 计算点云密度（单位：点/平方米）
    # 取X/Y方向的范围计算覆盖面积（简化为矩形面积）
    x_range = points[:, 0].max() - points[:, 0].min()
    y_range = points[:, 1].max() - points[:, 1].min()
    area = x_range * y_range if (x_range > 0 and y_range > 0) else 1.0  # 避免除以0
    density = n_points / area  # 点/平方米

    # 2. 判断是否需要降采样：点数超过阈值且密度超过阈值才降采样
    if n_points >= min_points and density >= min_density:
        step = max(1, int(step))
        downsampled_points = points[::step]
        if residuals is not None:
            downsampled_residuals = residuals[::step]
            print(f"降采样完成：原始点数={n_points}, 降采样后点数={len(downsampled_points)}, "
                  f"密度={density:.1f}点/㎡")
            return downsampled_points, downsampled_residuals
        print(f"降采样完成：原始点数={n_points}, 降采样后点数={len(downsampled_points)}, "
              f"密度={density:.1f}点/㎡")
        return downsampled_points
    else:
        # 不需要降采样，返回原始数据
        print(f"无需降采样：原始点数={n_points}, 密度={density:.1f}点/㎡ "
              f"(阈值：点数≥{min_points}且密度≥{min_density}点/㎡才降采样)")
        if residuals is not None:
            return points, residuals
        return points


def extract_outlier_ratio(file_name):
    """从文件名提取外点比例（适配"10pct"格式，10表示10%）"""
    import re
    # 匹配 "数字pct" 格式（如10pct、25pct），捕获数字部分
    match = re.search(r'(\d+)pct', file_name)
    if not match:
        raise ValueError(f"文件名 {file_name} 不包含外点比例信息（需包含类似'10pct'的标识）")
    # 提取数字并转换为比例（10pct → 10% → 0.1）
    percentage = int(match.group(1))  # 提取"10"并转为整数
    return percentage / 100.0  # 转换为比例（如10 → 0.1）


def parse_array_from_string(s, dtype=float):
    """安全解析CSV中的数组字符串"""
    if not isinstance(s, str) or s.strip() in ["None", "", "[]"]:
        return None
    try:
        clean_str = s.strip().strip('[]')
        if not clean_str:
            return None

        if dtype == bool:
            # 布尔特殊处理：支持 'True', 'False', '1', '0'
            parts = [x.strip().capitalize() for x in clean_str.split(',')]
            # 处理无逗号情况（空格分隔）
            if len(parts) == 1 and ' ' in clean_str:
                parts = [x.strip().capitalize() for x in clean_str.split()]

            bool_list = []
            for p in parts:
                if p in ('True', '1'):
                    bool_list.append(True)
                elif p in ('False', '0'):
                    bool_list.append(False)
                else:
                    raise ValueError(f"无法解析布尔值: {p}")
            return np.array(bool_list, dtype=bool)
        else:
            # 数值类型：兼容逗号和空格
            if ',' in clean_str:
                parts = [x.strip() for x in clean_str.split(',')]
                clean_str = ' '.join(parts)
            return np.fromstring(clean_str, dtype=dtype, sep=' ')
    except Exception as e:
        print(f"解析失败: {s} | 错误: {e}")
        return None
