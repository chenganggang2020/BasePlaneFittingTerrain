import numpy as np


def calculate_detection_metrics(true_labels, predicted_inliers):
    """计算检测指标（准确率、假阳性率等）"""
    if predicted_inliers is None or len(predicted_inliers) == 0:
        return {
            "Accuracy": np.nan,
            "False Negative Rate": np.nan,
            "False Positive Rate": np.nan
        }

    # 核心修复：将列表转换为 numpy 布尔数组
    if isinstance(predicted_inliers, list):
        # 若 inliers 是索引列表（如 [0,1,2]），先转换为布尔数组
        if len(predicted_inliers) > 0 and isinstance(predicted_inliers[0], int):
            predicted_inliers = np.isin(np.arange(len(true_labels)), predicted_inliers)
        else:
            # 若已是布尔列表，直接转换
            predicted_inliers = np.array(predicted_inliers, dtype=bool)
    elif not isinstance(predicted_inliers, np.ndarray) or predicted_inliers.dtype != bool:
        # 确保是布尔数组
        predicted_inliers = np.array(predicted_inliers, dtype=bool)

    # 后续计算（此时 ~predicted_inliers 合法）
    tp = np.sum(true_labels & predicted_inliers)  # 真阳性
    tn = np.sum(~true_labels & ~predicted_inliers)  # 真阴性（关键：~ 操作仅对 numpy 数组有效）
    fp = np.sum(~true_labels & predicted_inliers)  # 假阳性
    fn = np.sum(true_labels & ~predicted_inliers)  # 假阴性

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan  # 假阳性率
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan  # 假阴性率
    return {
        "Accuracy": accuracy,
        "False Negative Rate": fnr,
        "False Positive Rate": fpr
    }


def calculate_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """计算两个向量之间的最小夹角（结果在 0 到 90 度之间）"""
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    if v1_norm == 0 or v2_norm == 0:
        return 0.0  # 零向量夹角定义为 0

    dot_product = np.dot(v1, v2)
    # 计算余弦值，abs() 确保了余弦值非负，对应的角度在 [0, 90] 度
    cos_theta = np.abs(dot_product) / (v1_norm * v2_norm)
    # 由于 cos_theta 已经是非负的，arccos 的结果将在 [0, pi/2] 弧度之间
    angle_rad = np.arccos(np.clip(cos_theta, 0.0, 1.0))

    return np.degrees(angle_rad)

# 在您的代码中添加这个函数（通常放在文件顶部或相关函数附近）
def calculate_d_est_difference(d_est, initial_d_est=0.0):
    """计算当前平面常数项 d_est 与初始基准面常数项的绝对差值"""
    return abs(d_est - initial_d_est)

