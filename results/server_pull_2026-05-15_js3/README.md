# js3 Blockelite Server Pull - 2026-05-15

Source server: `root@js3.blockelite.cn:21728`

Remote source directory: `/root/timeshadow`

Included:
- `timeshadow/results/`: channel probe JSON result files
- `timeshadow/logs/`: execution and download logs
- `timeshadow/analysis/`: rerun plan, audit notes, and audit script/output
- `timeshadow/harness/`: scripts used to run channel probe experiments

Excluded:
- `/root/timeshadow/models/` (large model weights, about 77 GB)
- `/root/timeshadow/.venv/` (environment artifacts)
- `/root/timeshadow/adapters/` (empty at sync time)
- `/root/timeshadow/hf_home/` (cache/runtime artifacts)

This pull preserves the remote experiment outputs without overwriting existing project results.
