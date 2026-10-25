# Three-policy evaluation and data export

This workflow reuses the final Asym_PPO, DreamWaQ_CENet and PayloadEstimator
checkpoints listed in `scripts/migrate_payload_comparison.py`. It writes new
results without updating training code, deployment parameters, or weights.

The scripts have separate purposes:

- `scripts/payload_metrics.py`: trial metrics and summaries shared by both datasets.
- `scripts/analyze_payload_benchmark.py`: audit and recompute all 540 saved Isaac
  Lab trials from the `task1_v2` dataset; preserve legacy completion semantics.
- `scripts/evaluate_payload_mujoco.py`: new synchronous flat-ground trials using
  the checkpoint actors, three added masses, and paired initial conditions.
- `scripts/report_payload_evaluation.py`: compact CSV tables, PNG/PDF figures,
  metric definitions, a Chinese README and optional data archive.

Choose a fresh results directory. For example, from the repository root:

```bash
task_results="$PWD/results/payload_comparison/evaluation_new"
mkdir -p "$task_results"
PYTHONDONTWRITEBYTECODE=1 /home/hyc/miniconda3/envs/env_parkour_deploy/bin/python \
  scripts/analyze_payload_benchmark.py \
  --input-root /home/hyc/dreamwaq_go2/logs/payload_benchmark/2026-10-25_unified_v1 \
  --output-dir "$task_results/isaaclab_recomputed"

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="/home/hyc/miniconda3/envs/env_parkour_deploy/lib/python3.11/site-packages:${PYTHONPATH:-}" \
PYTHONEXE=/home/hyc/miniconda3/envs/env_isaaclab/bin/python \
/home/hyc/isaac-sim/python.sh scripts/evaluate_payload_mujoco.py \
  --output-dir "$task_results/mujoco_fresh"

PYTHONDONTWRITEBYTECODE=1 /home/hyc/miniconda3/envs/env_unitree/bin/python \
  scripts/report_payload_evaluation.py --results-root "$task_results"
PYTHONDONTWRITEBYTECODE=1 /home/hyc/miniconda3/envs/env_unitree/bin/python \
  scripts/report_payload_evaluation.py --results-root "$task_results" --package-only
```

The local launcher and existing environments provide dependencies; no Isaac Sim
application or GPU is started by the fresh MuJoCo evaluation. The extra Python path
provides the existing Python 3.11 MuJoCo package. The simulator's XML is read only.
New MuJoCo trials modify the loaded model's base mass and scale inertia in memory.

The existing Isaac Lab dataset comprises 3 terrains × 3 masses × 2 commands ×
3 models × 10 initial conditions = 540 trials, each 12 seconds including 2 seconds
of unscored warmup. Fresh MuJoCo defaults to flat ground, the same masses/commands,
and 5 paired initial conditions = 90 trials. Its 12-second policy rollout follows
one second of PD settling and includes two seconds of unscored warmup.

Primary metrics are body-frame XY velocity-vector RMSE and yaw-rate RMSE.
Posture outputs include mean, RMS, within-trial SD and peak absolute roll/pitch;
SD/mean/peak use degrees, while legacy RMS fields retain radians. Mass error uses
added mass in kg and is unavailable for the two baselines. Summaries give trial
means, sample SD and sample counts. Early failures remain in the data; trials
without scored samples retain failure/time fields and blank scored metrics.

The Isaac Lab legacy completion rate still uses survival and maximum world-x
progress of 3.2 m. Interpret turning separately; this is not a new trajectory-based
success definition. The supplementary planar reference integrates commands from
initial yaw and ignores terrain pitch. Fresh MuJoCo has a documented fall criterion
and no task-completion threshold. Do not pool the two engines, or identify lower
posture fluctuation alone as better stability when progress differs.

Only one training seed per method is available. Paired initial conditions measure
test variation and do not establish repeatability over training seeds. These are
simulated measurements, not real-robot or asynchronous C++/DDS measurements.

Generated data, raw traces, figures and archives remain under the ignored `results/`
directory. Source hashes in manifests allow verification that inputs were preserved.
