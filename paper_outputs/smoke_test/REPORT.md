# COSMOS paper pipeline report

## How to reproduce

Command run:

`/workspace/paper_pipeline.py --plan /workspace/paper_plan_smoke.yaml --resume --outdir /workspace/paper_outputs/smoke_test --workers 1`

## Plan summary

- **plan**: `/workspace/paper_plan_smoke.yaml`
- **plan_hash_sha256**: `d313d35196df4f63d66249fc4ebd6bb3a67d7611734c5b036b3b77a0b7a1390a`
- **seeds**: 2 (seed values: 0..1)
- **tmax**: 20.0
- **max_events**: 2000
- **enable_bb**: False

## EXACT JOBS EXECUTED

- `baseline/smooth/uniform`
- `baseline/smooth/concentrated`
- `intelligence/smooth/uniform`
- `baseline/rugged/uniform`
- `intelligence/rugged/uniform`
- `baseline/cluster/uniform`
- `baseline/cluster/concentrated`
- `intelligence/cluster/uniform`
- `cluster_suite/cluster/uniform`

### Required jobs list (from pipeline)

- `baseline/smooth/uniform`
- `baseline/smooth/concentrated`
- `intelligence/smooth/uniform`
- `baseline/rugged/uniform`
- `intelligence/rugged/uniform`
- `baseline/cluster/uniform`
- `baseline/cluster/concentrated`
- `intelligence/cluster/uniform`

- **All required jobs present: YES**

## Runtime summary

- **total wall time**: 00:01
- **workers used**: 1
- **avg per run (including failures)**: 00:01

## Failures summary

- **n_fail**: 0
- See `/workspace/paper_outputs/smoke_test/pipeline.log` and per-run `raw/<job>/seed_<seed>/run.log`.

## W_long definition (by job)

W_long is defined as the W component corresponding to the vacuum with the largest tau
in the topology configuration.

- `baseline/smooth/uniform` -> long_index=2 taus=[1.0, 10.0, 100.0]
- `baseline/smooth/concentrated` -> long_index=2 taus=[1.0, 10.0, 100.0]
- `intelligence/smooth/uniform` -> long_index=2 taus=[1.0, 10.0, 100.0]
- `baseline/rugged/uniform` -> long_index=2 taus=[1.0, 10.0, 100.0]
- `intelligence/rugged/uniform` -> long_index=2 taus=[1.0, 10.0, 100.0]
- `baseline/cluster/uniform` -> long_index=2 taus=[1.0, 10.0, 100.0]
- `baseline/cluster/concentrated` -> long_index=2 taus=[1.0, 10.0, 100.0]
- `intelligence/cluster/uniform` -> long_index=2 taus=[1.0, 10.0, 100.0]
- `cluster_suite/cluster/uniform` -> long_index=2 taus=[1.0, 10.0, 100.0]

## Key numeric results

- **smooth / baseline**: mean final_W_long=0.447172 (95% CI [0.438658, 0.455686]), extinction_rate=0.0000
- **smooth / intelligence**: mean final_W_long=0.497951 (95% CI [0.461114, 0.534788]), extinction_rate=0.0000
- **rugged / baseline**: mean final_W_long=0.677248 (95% CI [0.660764, 0.693732]), extinction_rate=0.0000
- **rugged / intelligence**: mean final_W_long=0.525052 (95% CI [0.475962, 0.574141]), extinction_rate=0.0000
- **cluster / baseline**: mean final_W_long=0.499525 (95% CI [0.491884, 0.507167]), extinction_rate=0.0000
- **cluster / intelligence**: mean final_W_long=0.622333 (95% CI [0.595978, 0.648689]), extinction_rate=0.0000

## Outputs

- **run index**: `/workspace/paper_outputs/smoke_test/tables/run_index.csv`
- **job summary**: `/workspace/paper_outputs/smoke_test/tables/job_summary.csv`
- **figures**: `/workspace/paper_outputs/smoke_test/figures` (PNG)
- **log**: `/workspace/paper_outputs/smoke_test/pipeline.log`
