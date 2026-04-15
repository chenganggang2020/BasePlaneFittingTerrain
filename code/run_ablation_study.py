import argparse
import os

from experiment_presets import (
    ABLATION_CASES,
    dump_manifest,
    get_ablation_case,
    get_paper_preset,
    run_ablation_case,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ablation studies for the paper-style pipeline.")
    parser.add_argument(
        "--preset",
        default="peiv35",
        help="Paper preset to use as the dataset and geometry baseline.",
    )
    parser.add_argument(
        "--ablation",
        default="all",
        help="Ablation case name to run. Use 'all' to run every registered case.",
    )
    parser.add_argument(
        "--workers",
        default=1,
        help="Number of worker processes for point-cloud-level parallelism. Use 0 or 'auto' for CPU-count-based auto mode.",
    )
    return parser.parse_args()


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    args = parse_args()
    preset = get_paper_preset(args.preset)
    ablation_names = list(ABLATION_CASES.keys()) if args.ablation == "all" else [args.ablation]

    manifest = []
    for ablation_name in ablation_names:
        case = get_ablation_case(ablation_name)
        run_ablation_case(
            args.preset,
            ablation_name,
            run_overrides={"max_workers": args.workers},
        )
        manifest.append({
            "preset": args.preset,
            "ablation": ablation_name,
            "workers": args.workers,
            "description": case.description,
            "output_root": os.path.abspath(os.path.join(preset.output_root, "ablations", case.output_suffix)),
            "analyzer_overrides": case.analyzer_overrides,
            "run_overrides": case.run_overrides,
        })

    manifest_path = os.path.abspath(os.path.join(preset.output_root, "ablations", "ablation_manifest.json"))
    dump_manifest(manifest_path, manifest)
    print(f"\nAblation manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
