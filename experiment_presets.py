from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from matplotlib import rcParams

from monte_carlo import PointCloudAnalyzer


def configure_plotting():
    rcParams.update({
        "text.antialiased": True,
        "axes.edgecolor": "black",
        "axes.facecolor": "white",
        "font.family": ["SimHei", "Microsoft YaHei", "Arial", "sans-serif"],
        "font.size": 10,
        "lines.antialiased": True,
        "backend": "Agg",
    })


def ensure_runtime_dependencies():
    return None


@dataclass(frozen=True)
class ExperimentPreset:
    name: str
    point_cloud_dir: str
    output_root: str
    initial_plane_slope: float
    initial_plane_interception: float
    tau: float = 0.05
    eps: float = 0.5
    random_seed: int = 42
    force_reprocess: bool = True
    dpi: int = 1200
    save_vector: bool = True
    analyzer_overrides: dict[str, Any] = field(default_factory=dict)
    run_overrides: dict[str, Any] = field(default_factory=dict)

    def build_analyzer_kwargs(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        kwargs = {
            "tau": self.tau,
            "eps": self.eps,
            "initial_plane_slope": self.initial_plane_slope,
            "initial_plane_interception": self.initial_plane_interception,
        }
        kwargs.update(self.analyzer_overrides)
        if overrides:
            kwargs.update(overrides)
        return kwargs

    def build_run_kwargs(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        kwargs = {
            "point_cloud_dir": self.point_cloud_dir,
            "output_root": self.output_root,
            "random_seed": self.random_seed,
            "force_reprocess": self.force_reprocess,
            "dpi": self.dpi,
            "save_vector": self.save_vector,
        }
        kwargs.update(self.run_overrides)
        if overrides:
            kwargs.update(overrides)
        return kwargs


@dataclass(frozen=True)
class AblationCase:
    name: str
    description: str
    output_suffix: str
    analyzer_overrides: dict[str, Any] = field(default_factory=dict)
    run_overrides: dict[str, Any] = field(default_factory=dict)


PAPER_EXPERIMENT_PRESETS = OrderedDict({
    "peiv10": ExperimentPreset(
        name="peiv10",
        point_cloud_dir="point_clouds_output_peiv",
        output_root="results/paper_peiv10",
        initial_plane_slope=10,
        initial_plane_interception=30,
    ),
    "peiv15": ExperimentPreset(
        name="peiv15",
        point_cloud_dir="point_clouds_output_peiv15",
        output_root="results/paper_peiv15",
        initial_plane_slope=15,
        initial_plane_interception=20,
        force_reprocess=False,
    ),
    "peiv35": ExperimentPreset(
        name="peiv35",
        point_cloud_dir="point_clouds_output_peiv35",
        output_root="results/paper_peiv35",
        initial_plane_slope=35,
        initial_plane_interception=30,
    ),
    "structured35": ExperimentPreset(
        name="structured35",
        point_cloud_dir="point_clouds_output_structured35",
        output_root="results/paper_structured35",
        initial_plane_slope=35,
        initial_plane_interception=30,
    ),
})


ABLATION_CASES = OrderedDict({
    "full_pipeline": AblationCase(
        name="full_pipeline",
        description="Run all algorithms with risk, safety, and joint evaluation enabled.",
        output_suffix="full_pipeline",
    ),
    "no_mrf": AblationCase(
        name="no_mrf",
        description="Disable MRF-style smoothing in the macro risk map.",
        output_suffix="no_mrf",
        analyzer_overrides={"risk_mrf_lambda": 0.0, "risk_mrf_iters": 0},
    ),
    "no_joint": AblationCase(
        name="no_joint",
        description="Disable joint risk-safety fusion and keep separate risk/safety outputs only.",
        output_suffix="no_joint",
        analyzer_overrides={"enable_joint_analysis": False},
    ),
    "risk_only": AblationCase(
        name="risk_only",
        description="Run macro risk probability analysis only.",
        output_suffix="risk_only",
        analyzer_overrides={
            "enable_risk_analysis": True,
            "enable_safety_analysis": False,
            "enable_joint_analysis": False,
        },
    ),
    "safety_only": AblationCase(
        name="safety_only",
        description="Run method-wise safety evaluation only.",
        output_suffix="safety_only",
        analyzer_overrides={
            "enable_risk_analysis": False,
            "enable_safety_analysis": True,
            "enable_joint_analysis": False,
        },
    ),
    "classic_baselines": AblationCase(
        name="classic_baselines",
        description="Exclude the two newly added paper-style methods and keep classical baselines.",
        output_suffix="classic_baselines",
        analyzer_overrides={"exclude_methods": ["RobustLS-TLS", "RSTLS"]},
    ),
    "paper_methods_only": AblationCase(
        name="paper_methods_only",
        description="Focus on the main paper-relevant fitting methods.",
        output_suffix="paper_methods_only",
        analyzer_overrides={
            "include_methods": ["LS", "RANSAC", "RWTLS", "RobustLS-TLS", "RSTLS"],
        },
    ),
})


def get_paper_preset(name: str) -> ExperimentPreset:
    try:
        return PAPER_EXPERIMENT_PRESETS[name]
    except KeyError as exc:
        available = ", ".join(PAPER_EXPERIMENT_PRESETS.keys())
        raise KeyError(f"Unknown paper preset '{name}'. Available presets: {available}") from exc


def get_ablation_case(name: str) -> AblationCase:
    try:
        return ABLATION_CASES[name]
    except KeyError as exc:
        available = ", ".join(ABLATION_CASES.keys())
        raise KeyError(f"Unknown ablation case '{name}'. Available cases: {available}") from exc


def run_preset(
    preset: ExperimentPreset,
    analyzer_overrides: dict[str, Any] | None = None,
    run_overrides: dict[str, Any] | None = None,
):
    ensure_runtime_dependencies()
    configure_plotting()

    analyzer = PointCloudAnalyzer(**preset.build_analyzer_kwargs(analyzer_overrides))
    return analyzer.run_outlier_sweep(**preset.build_run_kwargs(run_overrides))


def run_named_paper_preset(name: str, analyzer_overrides: dict[str, Any] | None = None, run_overrides: dict[str, Any] | None = None):
    return run_preset(get_paper_preset(name), analyzer_overrides=analyzer_overrides, run_overrides=run_overrides)


def build_ablation_output_root(base_output_root: str, ablation_case: AblationCase) -> str:
    return os.path.join(base_output_root, "ablations", ablation_case.output_suffix)


def run_ablation_case(
    preset_name: str,
    ablation_name: str,
    run_overrides: dict[str, Any] | None = None,
    analyzer_overrides: dict[str, Any] | None = None,
):
    preset = get_paper_preset(preset_name)
    case = get_ablation_case(ablation_name)
    merged_run_overrides = dict(case.run_overrides)
    if run_overrides:
        merged_run_overrides.update(run_overrides)
    merged_run_overrides["output_root"] = build_ablation_output_root(preset.output_root, case)
    merged_analyzer_overrides = dict(case.analyzer_overrides)
    if analyzer_overrides:
        merged_analyzer_overrides.update(analyzer_overrides)
    return run_preset(
        preset,
        analyzer_overrides=merged_analyzer_overrides,
        run_overrides=merged_run_overrides,
    )


def dump_manifest(manifest_path: str, records: list[dict[str, Any]]):
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
