import argparse
import os
from multiprocessing import freeze_support

from OutlierTerrainSimulator import (
    build_default_structured_simulator_kwargs,
    generate_structured_dataset,
)
from experiment_presets import (
    ABLATION_CASES,
    dump_manifest,
    get_paper_preset,
    run_ablation_case,
    run_named_paper_preset,
)
from reproduce_paper_figures import reproduce_paper_artifacts


def infer_dem_dir(point_cloud_dir):
    if "point_clouds" in point_cloud_dir:
        return point_cloud_dir.replace("point_clouds", "dem_files")
    return f"{point_cloud_dir}_dem"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full structured point-cloud pipeline: generation, experiment, ablation, and paper-style reporting."
    )
    parser.add_argument("--preset", default="structured35")
    parser.add_argument(
        "--workers",
        default="auto",
        help="Number of worker processes for experiment execution. Use 0 or 'auto' for CPU-count-based auto mode.",
    )
    parser.add_argument(
        "--generation-workers",
        default=None,
        help="Number of worker processes for dataset generation. Defaults to --workers.",
    )
    parser.add_argument("--force-reprocess", action="store_true")
    parser.add_argument("--no-save-vector", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-paper-report", action="store_true")
    parser.add_argument("--ablation", default="all", help="Ablation case name or 'all'.")
    parser.add_argument("--point-cloud-dir", default=None)
    parser.add_argument("--dem-dir", default=None)
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
    return parser.parse_args()


def parse_method_list(value):
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


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
        "preset": args.preset,
        "point_cloud_dir": point_cloud_dir,
        "dem_dir": dem_dir,
        "workers": workers,
        "generation_workers": generation_workers,
        "dpi": args.dpi,
        "save_vector": save_vector,
        "force_reprocess": args.force_reprocess,
        "analyzer_overrides": base_analyzer_overrides,
    }

    if not args.skip_generation:
        simulator_kwargs = build_default_structured_simulator_kwargs(
            slope_angle=preset.initial_plane_slope,
            base_z=preset.initial_plane_interception,
        )
        generated_ratios = generate_structured_dataset(
            dem_dir=dem_dir,
            pc_dir=point_cloud_dir,
            ratio_start=args.ratio_start,
            ratio_stop=args.ratio_stop,
            ratio_step=args.ratio_step,
            simulator_kwargs=simulator_kwargs,
            max_workers=generation_workers,
        )
        manifest["generated_ratio_count"] = len(generated_ratios)

    main_run_overrides = {
        "point_cloud_dir": point_cloud_dir,
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

    main_output_root = os.path.abspath(preset.output_root)
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
