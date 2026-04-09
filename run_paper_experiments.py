import argparse
import os

from experiment_presets import (
    PAPER_EXPERIMENT_PRESETS,
    dump_manifest,
    get_paper_preset,
    run_named_paper_preset,
)
from reproduce_paper_figures import reproduce_paper_artifacts


def parse_args():
    parser = argparse.ArgumentParser(description="Run paper-style outlier sweep experiments.")
    parser.add_argument(
        "--preset",
        default="all",
        help="Preset name to run. Use 'all' to run every paper preset.",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Override the preset and force recomputation of cached outputs.",
    )
    parser.add_argument(
        "--no-save-vector",
        action="store_true",
        help="Disable PDF/vector figure export.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="Override output figure DPI.",
    )
    parser.add_argument(
        "--skip-paper-report",
        action="store_true",
        help="Skip the paper-style figure/table reproduction step.",
    )
    parser.add_argument(
        "--workers",
        default=1,
        help="Number of worker processes for point-cloud-level parallelism. Use 0 or 'auto' for CPU-count-based auto mode.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    preset_names = list(PAPER_EXPERIMENT_PRESETS.keys()) if args.preset == "all" else [args.preset]

    manifest = []
    for preset_name in preset_names:
        preset = get_paper_preset(preset_name)
        analyzer_overrides = {}
        run_overrides = {}
        if args.force_reprocess:
            run_overrides["force_reprocess"] = True
        if args.no_save_vector:
            run_overrides["save_vector"] = False
        if args.dpi is not None:
            run_overrides["dpi"] = args.dpi
        run_overrides["max_workers"] = args.workers

        run_named_paper_preset(
            preset_name,
            analyzer_overrides=analyzer_overrides,
            run_overrides=run_overrides,
        )
        output_root = os.path.abspath(run_overrides.get("output_root", preset.output_root))
        reproduction_output = None
        if not args.skip_paper_report:
            report = reproduce_paper_artifacts(
                preset_name=preset_name,
                results_root=output_root,
                save_vector=not args.no_save_vector,
            )
            reproduction_output = report["output_dir"]
        record = {
            "preset": preset_name,
            "point_cloud_dir": preset.point_cloud_dir,
            "output_root": output_root,
            "workers": args.workers,
            "force_reprocess": run_overrides.get("force_reprocess", preset.force_reprocess),
            "save_vector": run_overrides.get("save_vector", preset.save_vector),
            "dpi": run_overrides.get("dpi", preset.dpi),
            "paper_reproduction_output": reproduction_output,
        }
        manifest.append(record)

    manifest_path = os.path.abspath(os.path.join("results", "paper_experiment_manifest.json"))
    dump_manifest(manifest_path, manifest)
    print(f"\nPaper experiment manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
