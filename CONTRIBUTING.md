# Contributing

This repository is primarily organized as a research codebase, so the goal is to keep experiments reproducible and commits easy to review.

## Recommended Branch Workflow

Use short-lived branches from `main`, for example:

```bash
git checkout -b codex/improve-rstls-threshold
```

Good branch topics include:

- simulation updates
- algorithm changes
- plotting and reporting improvements
- experiment pipeline fixes

## Commit Style

Prefer small, focused commits with clear intent, for example:

```bash
git commit -m "Improve structured noise simulation"
git commit -m "Tighten RSTLS inlier refinement"
git commit -m "Polish paper-style global plots"
```

## Before Opening a PR

Please try to complete the relevant checks:

1. run a small smoke test for the changed pipeline path
2. make sure the modified scripts can be imported or compiled
3. avoid committing generated datasets or `results/`
4. update `README.md` when the user-facing workflow changes

## Platform Notes

- On Windows, use `python`.
- On Linux or Kylin, use `python3`.
- The pipeline is designed to fall back gracefully when some optional packages are unavailable.

## Large Outputs

Do not commit large generated artifacts such as:

- DEM outputs
- point clouds
- experiment result directories
- temporary debug folders

Share them through releases, Git LFS, or external storage instead.
