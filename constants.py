# 点云数量阈值（超过此值则不执行RWTLS）
RWTLS_MEMORY_THRESHOLD = 20000  # 可根据实际内存情况调整（单位：点）

# 算法颜色方案
ALGORITHM_COLORS = {
    # 随机算法
    "MAD": "#1f77b4",  # 深蓝色
    "LMedsq": "#ff7f0e",  # 橙色
    "RANSAC": "#7f7f7f",  # 深灰色
    "IKOSE": "#d62728",  # 深红色
    "PDIMSE": "#9467bd",  # 紫色
    # 确定性算法
    "LS": "#8c564b",  # 棕色
    "RobustLS-TLS": "#17becf",  # 青色
    "RWTLS": "#e377c2",  # 粉色（无数据时会调整为灰色）
    "RSTLS": "#2ca02c"  # 深绿色
}


# 算法标记配置
ALGORITHM_MARKERS = {
    "MAD": "o", "LMedsq": "s", "RANSAC": "D", "IKOSE": "^",
    "PDIMSE": "v", "LS": "P", "RobustLS-TLS": "X", "RWTLS": "H", "RSTLS": "*"
}



PLOT_CONFIG = {
    "figsize": (10, 6),
    "font_size": 12,
    "title_size": 12,
    "grid_style": '--',
    "grid_alpha": 0.7
}

