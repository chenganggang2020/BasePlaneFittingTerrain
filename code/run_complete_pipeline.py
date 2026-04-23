import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import freeze_support

from OutlierTerrainSimulator import (
    OutlierTerrainSimulator,
    build_point_cloud_filename,
    build_ratio_seed,
    build_ratio_values,
    build_default_structured_simulator_kwargs,
    generate_structured_dataset,
    simulation_outputs_exist,
)
from experiment_presets import (
    ABLATION_CASES,
    dump_manifest,
    get_paper_preset,
    run_ablation_case,
    run_named_paper_preset,
)
from monte_carlo import PointCloudAnalyzer
from reproduce_paper_figures import reproduce_paper_artifacts

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    class _TqdmFallback:
        def __init__(self, iterable=None, total=None, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter(())

        def update(self, n=1):
            return None

        def close(self):
            return None

    def tqdm(iterable=None, *args, **kwargs):
        return _TqdmFallback(iterable=iterable, **kwargs)


def infer_dem_dir(point_cloud_dir):
    if "point_clouds" in point_cloud_dir:
        return point_cloud_dir.replace("point_clouds", "dem_files")
    return f"{point_cloud_dir}_dem"


def _load_json_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Pipeline config must be a JSON object: {config_path}")
    return payload


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run the full structured point-cloud pipeline: generation, experiment, ablation, and paper-style reporting."
    )
    parser.add_argument("--config", default=None, help="Optional JSON config file.")
    parser.add_argument("--preset", default="structured35")
    parser.add_argument(
        "--pipeline-mode",
        default="streaming",
        choices=["streaming", "staged"],
        help="Use 'streaming' to generate and analyze each ratio in one parallel workflow, or 'staged' to keep generation and analysis as separate steps.",
    )
    parser.add_argument(
        "--workers",
        default="4",
        help="Number of worker processes for experiment execution. Use 0 or 'auto' for CPU-count-based auto mode.",
    )
    parser.add_argument(
        "--generation-workers",
        default=None,
        help="Number of worker processes for dataset generation. Defaults to --workers.",
    )
    parser.add_argument(
        "--simulator-profile",
        default="landing",
        choices=["baseline", "landing", "stress"],
        help="Structured terrain profile used during synthetic data generation.",
    )
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--force-reprocess", action="store_true")
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Regenerate point clouds even if matching synthetic files already exist.",
    )
    parser.add_argument("--no-save-vector", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-paper-report", action="store_true")
    parser.add_argument("--ablation", default="all", help="Ablation case name or 'all'.")
    parser.add_argument("--point-cloud-dir", default=None)
    parser.add_argument("--dem-dir", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--ratio-start", type=float, default=0.10)
    parser.add_argument("--ratio-stop", type=float, default=0.91)
    parser.add_argument("--ratio-step", type=float, default=0.01)
    parser.add_argument(
        "--include-methods",
        default=None,
        help="Comma-separated algorithm list to keep, for example RANSAC,RobustLS-TLS,RSTLS.",
    )
    parser.add_argument(
        "--exclude-methods",
        default=None,
        help="Comma-separated algorithm list to exclude.",
    )
    parser.add_argument("--disable-risk", action="store_true")
    parser.add_argument("--disable-safety", action="store_true")
    parser.add_argument("--disable-joint", action="store_true")
    return parser


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, _ = pre_parser.parse_known_args()

    parser = build_arg_parser()
    config = None
    if pre_args.config:
        config = _load_json_config(pre_args.config)
        valid_keys = {action.dest for action in parser._actions}
        unknown_keys = sorted(set(config.keys()) - valid_keys)
        if unknown_keys:
            raise ValueError(
                f"Unknown config keys in {pre_args.config}: {', '.join(unknown_keys)}"
            )
        parser.set_defaults(**config)

    args = parser.parse_args()
    args.config_data = config if pre_args.config else None
    return args


def parse_method_list(value):
    if not value:
        return None
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_max_workers(max_workers):
    if max_workers is None:
        return 1
    if isinstance(max_workers, str):
        if max_workers.lower() == "auto":
            max_workers = 0
        else:
            max_workers = int(max_workers)
    if int(max_workers) <= 0:
        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count - 1)
    return max(1, int(max_workers))


def _build_parallel_executor(max_workers):
    try:
        return ProcessPoolExecutor(max_workers=max_workers), "process"
    except (PermissionError, OSError) as exc:
        print(f"Warning: process pool unavailable ({exc}); falling back to thread pool.")
        return ThreadPoolExecutor(max_workers=max_workers), "thread"


def _generate_and_analyze_ratio_worker(simulator_kwargs, analyzer_kwargs, task):
    ratio = task["ratio"]
    if not task["force_regenerate"] and simulation_outputs_exist(task["dem_dir"], task["point_cloud_dir"], ratio):
        generation_record = {
            "outlier_ratio": float(ratio),
            "point_cloud_path": os.path.join(task["point_cloud_dir"], build_point_cloud_filename(ratio)),
            "skipped": True,
        }
    else:
        worker_kwargs = dict(simulator_kwargs)
        worker_kwargs["random_state"] = build_ratio_seed(task["base_seed"], task["ratio_index"], ratio)
        generation_record = OutlierTerrainSimulator(**worker_kwargs).generate_simulation(
            outlier_ratio=ratio,
            dem_dir=task["dem_dir"],
            pc_dir=task["point_cloud_dir"],
        )

    analyzer = PointCloudAnalyzer(**analyzer_kwargs)
    result_rows = analyzer.analyze_point_cloud(
        file_name=build_point_cloud_filename(ratio),
        outlier_ratio=ratio,
        eps=ratio,
        point_cloud_dir=task["point_cloud_dir"],
        output_root=task["output_root"],
        force_reprocess=task["force_reprocess"],
        dpi=task["dpi"],
    )
    return {
        "ratio": float(ratio),
        "generation": generation_record,
        "result_rows": result_rows,
    }


def run_streaming_generation_and_experiment(
    preset,
    point_cloud_dir,
    dem_dir,
    ratio_start,
    ratio_stop,
    ratio_step,
    simulator_profile,
    base_seed,
    analyzer_overrides,
    output_root,
    workers,
    force_reprocess,
    force_regenerate,
    dpi,
    save_vector,
):
    ratio_values = build_ratio_values(ratio_start=ratio_start, ratio_stop=ratio_stop, ratio_step=ratio_step)
    if not ratio_values:
        print("No ratios scheduled for streaming generation.")
        return []

    simulator_kwargs = build_default_structured_simulator_kwargs(
        slope_angle=preset.initial_plane_slope,
        base_z=preset.initial_plane_interception,
        profile=simulator_profile,
        random_state=base_seed,
    )
    analyzer_kwargs = preset.build_analyzer_kwargs(analyzer_overrides)
    analyzer = PointCloudAnalyzer(**analyzer_kwargs)
    os.makedirs(output_root, exist_ok=True)

    worker_count = min(_resolve_max_workers(workers), len(ratio_values))
    tasks = [
        {
            "ratio_index": index,
            "ratio": ratio,
            "dem_dir": dem_dir,
            "point_cloud_dir": point_cloud_dir,
            "output_root": output_root,
            "dpi": dpi,
            "base_seed": base_seed,
            "force_reprocess": force_reprocess,
            "force_regenerate": force_regenerate,
        }
        for index, ratio in enumerate(ratio_values)
    ]

    result_rows_collection = []
    if worker_count <= 1:
        for task in tqdm(tasks, desc="Streaming generate+solve"):
            record = _generate_and_analyze_ratio_worker(simulator_kwargs, analyzer_kwargs, task)
            result_rows_collection.append(record["result_rows"])
    else:
        executor, backend = _build_parallel_executor(worker_count)
        print(f"Streaming generation+analysis with {worker_count} {backend} workers.")
        with executor:
            future_to_ratio = {
                executor.submit(_generate_and_analyze_ratio_worker, simulator_kwargs, analyzer_kwargs, task): task["ratio"]
                for task in tasks
            }
            progress = tqdm(total=len(tasks), desc="Streaming generate+solve")
            try:
                for future in as_completed(future_to_ratio):
                    ratio = future_to_ratio[future]
                    try:
                        record = future.result()
                        result_rows_collection.append(record["result_rows"])
                    except Exception as exc:
                        print(f"Streaming worker failed for ratio {ratio:.2f}: {exc}")
                    progress.update(1)
            finally:
                progress.close()

    analyzer.finalize_outlier_sweep(
        point_cloud_dir=point_cloud_dir,
        output_root=output_root,
        result_rows_collection=result_rows_collection,
        dpi=dpi,
        save_vector=save_vector,
    )
    return ratio_values


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    args = parse_args()
    preset = get_paper_preset(args.preset)

    point_cloud_dir = args.point_cloud_dir or preset.point_cloud_dir
    dem_dir = args.dem_dir or infer_dem_dir(point_cloud_dir)
    workers = args.workers
    generation_workers = args.generation_workers if args.generation_workers is not None else workers
    save_vector = not args.no_save_vector
    include_methods = parse_method_list(args.include_methods)
    exclude_methods = parse_method_list(args.exclude_methods)

    base_analyzer_overrides = {}
    if include_methods is not None:
        base_analyzer_overrides["include_methods"] = include_methods
    if exclude_methods is not None:
        base_analyzer_overrides["exclude_methods"] = exclude_methods
    if args.disable_risk:
        base_analyzer_overrides["enable_risk_analysis"] = False
    if args.disable_safety:
        base_analyzer_overrides["enable_safety_analysis"] = False
    if args.disable_joint:
        base_analyzer_overrides["enable_joint_analysis"] = False

    manifest = {
        "config_path": os.path.abspath(args.config) if args.config else None,
        "preset": args.preset,
        "point_cloud_dir": point_cloud_dir,
        "dem_dir": dem_dir,
        "workers": workers,
        "generation_workers": generation_workers,
        "pipeline_mode": args.pipeline_mode,
        "simulator_profile": args.simulator_profile,
        "base_seed": args.base_seed,
        "dpi": args.dpi,
        "save_vector": save_vector,
        "force_reprocess": args.force_reprocess,
        "force_regenerate": args.force_regenerate,
        "analyzer_overrides": base_analyzer_overrides,
    }
    if getattr(args, "config_data", None):
        manifest["config_overrides"] = args.config_data

    main_output_root = os.path.abspath(args.output_root or preset.output_root)
    generated_ratios = []
    if not args.skip_generation and args.pipeline_mode == "streaming":
        generated_ratios = run_streaming_generation_and_experiment(
            preset=preset,
            point_cloud_dir=point_cloud_dir,
            dem_dir=dem_dir,
            ratio_start=args.ratio_start,
            ratio_stop=args.ratio_stop,
            ratio_step=args.ratio_step,
            simulator_profile=args.simulator_profile,
            base_seed=args.base_seed,
            analyzer_overrides=base_analyzer_overrides,
            output_root=main_output_root,
            workers=workers,
            force_reprocess=args.force_reprocess,
            force_regenerate=args.force_regenerate,
            dpi=args.dpi,
            save_vector=save_vector,
        )
    else:
        if not args.skip_generation:
            simulator_kwargs = build_default_structured_simulator_kwargs(
                slope_angle=preset.initial_plane_slope,
                base_z=preset.initial_plane_interception,
                profile=args.simulator_profile,
                random_state=args.base_seed,
            )
            generated_ratios = generate_structured_dataset(
                dem_dir=dem_dir,
                pc_dir=point_cloud_dir,
                ratio_start=args.ratio_start,
                ratio_stop=args.ratio_stop,
                ratio_step=args.ratio_step,
                simulator_kwargs=simulator_kwargs,
                max_workers=generation_workers,
                skip_existing=not args.force_regenerate,
            )

        main_run_overrides = {
            "point_cloud_dir": point_cloud_dir,
            "output_root": main_output_root,
            "max_workers": workers,
            "dpi": args.dpi,
            "save_vector": save_vector,
        }
        if args.force_reprocess:
            main_run_overrides["force_reprocess"] = True

        run_named_paper_preset(
            args.preset,
            analyzer_overrides=base_analyzer_overrides,
            run_overrides=main_run_overrides,
        )

    manifest["generated_ratio_count"] = len(generated_ratios)
    manifest["main_output_root"] = main_output_root

    if not args.skip_paper_report:
        report = reproduce_paper_artifacts(
            preset_name=args.preset,
            results_root=main_output_root,
            save_vector=save_vector,
        )
        manifest["paper_report_output"] = report["output_dir"]

    ablation_records = []
    if not args.skip_ablation:
        ablation_names = list(ABLATION_CASES.keys()) if args.ablation == "all" else [args.ablation]
        for ablation_name in ablation_names:
            run_ablation_case(
                args.preset,
                ablation_name,
                analyzer_overrides=base_analyzer_overrides,
                run_overrides={
                    "output_root": main_output_root,
                    "point_cloud_dir": point_cloud_dir,
                    "max_workers": workers,
                    "dpi": args.dpi,
                    "save_vector": save_vector,
                    **({"force_reprocess": True} if args.force_reprocess else {}),
                },
            )
            ablation_records.append({
                "ablation": ablation_name,
                "output_root": os.path.abspath(os.path.join(main_output_root, "ablations", ABLATION_CASES[ablation_name].output_suffix)),
            })
    manifest["ablations"] = ablation_records

    manifest_path = os.path.abspath(os.path.join(main_output_root, "full_pipeline_manifest.json"))
    dump_manifest(manifest_path, [manifest])
    print(f"\nFull pipeline manifest saved to: {manifest_path}")


if __name__ == "__main__":
    freeze_support()
    main()
