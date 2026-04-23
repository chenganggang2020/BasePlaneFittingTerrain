# Pipeline Configs

Use these JSON files as reusable experiment launch templates.

Examples:

```powershell
python code/run_complete_pipeline.py --config configs/pipeline_structured35.json
```

```powershell
python code/run_complete_pipeline.py --config configs/pipeline_smoke.json
```

Notes:

- Any value in the config can still be overridden from the command line.
- Config keys must match the command-line argument names without the leading `--`.
- Lists such as `include_methods` can be written as JSON arrays.
- `pipeline_smoke.json` now writes to isolated `data/..._smoke` and `results/debug/...` paths so it does not overwrite formal paper outputs.
