# algorithm_runner.py

from ls import LSPlaneFitter
from wtls import WTLSPlaneFitter
from mad import MADPlaneFitter
from pdimse import PDIMSEPlaneFitter
from lmedsq import LMedsqPlaneFitter
from ransac import RANSACPlaneFitter
from ikose import IKOSEPlaneFitter
from robust_ls_tls import RobustLSTLSPlaneFitter
from rwtls import RWTLSPlaneFitter
from sequential_robust_tls import SequentialRobustTLSPlaneFitter


def _filter_methods(methods, include_methods=None, exclude_methods=None):
    include_set = set(include_methods) if include_methods else None
    exclude_set = set(exclude_methods) if exclude_methods else set()

    filtered = []
    for method in methods:
        name = method[0]
        if include_set is not None and name not in include_set:
            continue
        if name in exclude_set:
            continue
        filtered.append(method)
    return filtered


def get_algorithm_categories(tau=1.2, eps=0.5, include_methods=None, exclude_methods=None):
    """
    返回确定性与随机性平面拟合算法列表，每个算法以 (name, callable, params_dict) 形式组织。

    :param tau: RANSAC 内点阈值（单位：米）
    :param eps: 初始外点比例估计（用于 RANSAC / PDIMSE 等）
    :return: dict 包含 stochastic_methods, deterministic_methods 等
    """

    # 随机性算法（需多次运行）
    stochastic_methods = [
        ("MAD", lambda pts, **kw: MADPlaneFitter(**kw).fit(pts), {}),
        ("LMedsq", lambda pts, **kw: LMedsqPlaneFitter(**kw).fit(pts), {"sample_num": 1000}),
        ("RANSAC", lambda pts, **kw: RANSACPlaneFitter(**kw).fit(pts), {
            "tau": tau,
            "p": 0.95,
            "eps": eps,
            "max_sample_num": 15000
        }),
        ("IKOSE", lambda pts, **kw: IKOSEPlaneFitter(**kw).fit(pts), {
            "K_ratio": 0.5,
            "max_iter": 100,
            "tol": 1e-3
        }),
        ("PDIMSE", lambda pts, **kw: PDIMSEPlaneFitter(**kw).fit(pts), {
            "alpha": 0.95,
            "P": 0.95,
            "eps": eps,
            "K_ratio": 0.2,
            "max_iter": 100,
            "tol": 1e-3
        })
    ]

    # 确定性算法（仅需运行一次）
    deterministic_methods = [
        ("LS", lambda pts, **kw: LSPlaneFitter().fit(pts), {}),
        ("RobustLS-TLS", lambda pts, **kw: RobustLSTLSPlaneFitter(**kw).fit(pts), {
            "ransac_tau": min(tau, 0.2),
            "ransac_max_sample_num": 2000,
            "max_iter": 50,
            "tol": 1e-5,
            "huber_c": 1.345,
            "adaptive_c": True,
            "min_c": 0.5,
            "sparse_eta": 0.25,
            "weight_decay": 0.6,
            "random_state": 42
        }),
        # ("WTLS", lambda pts, **kw: WTLSPlaneFitter(**kw).fit(pts), {
        #     "sigma_x": 1.0,
        #     "sigma_y": 1.0,
        #     "sigma_z": 1.0,
        #     "max_iter": 10,
        #     "eps_xi": 1e-5
        # }),
        ("RWTLS", lambda pts, **kw: RWTLSPlaneFitter(**kw).fit(pts), {
            "sigma_x": 1.0,
            "sigma_y": 1.0,
            "sigma_z": 1.0,
            "max_iter": 100,
            "eps_xi": 1e-5
        }),
        ("RSTLS", lambda pts, **kw: SequentialRobustTLSPlaneFitter(**kw).fit(pts), {
            "sigma_x": 0.05,
            "sigma_y": 0.05,
            "sigma_z": 0.08,
            "max_iter": 30,
            "eps_xi": 1e-6,
            "batch_size": 128,
            "ransac_tau": min(tau, 0.2),
            "ransac_max_sample_num": 800,
            "prior_precision": 1e-6,
            "batch_k1": 1.7,
            "batch_k2": 3.0,
            "hard_threshold_scale": 1.8,
            "hard_sigma_scale": 2.2,
            "hard_trim_quantile": 0.80,
            "hard_refit_iters": 3,
            "min_inlier_ratio": 0.18,
            "spatial_sort": True,
            "random_state": 42
        })
    ]

    stochastic_methods = _filter_methods(
        stochastic_methods,
        include_methods=include_methods,
        exclude_methods=exclude_methods,
    )
    deterministic_methods = _filter_methods(
        deterministic_methods,
        include_methods=include_methods,
        exclude_methods=exclude_methods,
    )

    sto_names = [m[0] for m in stochastic_methods]
    det_names = [m[0] for m in deterministic_methods]

    return {
        "stochastic_methods": stochastic_methods,
        "deterministic_methods": deterministic_methods,
        "sto_names": sto_names,
        "det_names": det_names,
        "all_methods": sto_names + det_names
    }
