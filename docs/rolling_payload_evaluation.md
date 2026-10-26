# Rolling sphere payload evaluation

This independent MuJoCo test reuses the original three final checkpoints and
the Oracle robot's tray/sphere geometry. Only evaluation code in
`/home/hyc/unitree_rl_lab` is changed; the training repository, `unitree_mujoco`
and their original results remain read only. No retraining, mass-input
interventions, reward changes or modifications to the controller are involved.

New files:

- `scripts/rolling_payload_support.py`: evaluation scene copies, physics checks,
  term-major history, loss monitoring and per-trial metrics.
- `scripts/evaluate_rolling_payload.py`: strict saved-policy loading via the
  existing migration helper, common physical preparation and paired rollouts.
- `scripts/report_rolling_payload.py`: raw-data audit, condition mean/sample SD,
  relative improvements, anomaly records and PNG/PDF figures.
- `scripts/test_rolling_payload.py`: meaningful physics, history, metric and
  cargo-loss regression checks.

Protocol fixed before formal rollouts:

- Flat `go2_Oracle/scene.xml`; sphere masses 2, 4, 6, 8 kg, radius 0.04 m.
- Base mass stays **6.921 kg**. The robot is 15.206408 kg plus sphere mass.
  The existing four tray walls add no separate mass/inertia because the base
  has an explicit inertial element. Geometry/contact settings are preserved.
- Each mass is set in a new XML **before compilation**, so the sphere inertia
  is `2/5*m*r^2`; there are no runtime mass edits or added external forces.
- The sphere keeps its free joint, with no weld, slider or actuator. Correct its
  initial overlap: center starts 0.099 m above base origin (top 0.057 m + radius
  0.04 m + 0.002 m clearance). Fix asset/include paths only in scene copies.
- Ten paired initial states, evaluation base seed **2026102601**, disjoint from
  previous seed sets. XY +/-[0.05,0.08]m, roll/pitch +/-0.02rad, yaw +/-0.1rad,
  joint offset/velocity +/-0.02. These random initial states are the same for
  all three models, with the same physical preparation at every mass.
- Before any actor, common PD stand for 2s, with a free sphere. Save the entire
  `mjSTATE_INTEGRATION` state once per mass/trial, then restore it exactly for
  every policy. Reject invalid common preparation; never select initial states
  by any policy's performance. The sphere must remain carried with robot contact
  in >=5 of the final 10 preparation samples.
- Final `model_4999.pt`, strict weights, saved architecture, actor eval/mean,
  input dimensions 45/270/1170, term-major history oldest to newest. True sphere
  mass, position and velocity are diagnostics only, never policy inputs.
- Commands `[0.6,0,0]`, scale 0.25, PD25/0.5, control50Hz. Preserve native MuJoCo
  physics step0.002s and actuator limits/friction. Robot foot friction is the
  source XML's0.4, sphere/wall sliding friction2.0. This is **not** the previous
  Isaac Lab1.0-friction experiment; numerical results must remain separate.
- 12s per policy rollout; first2s transition. Post-step samples on intervals
  beginning >=2s score `(2,12]s`, 500 samples for a complete carried rollout.
- Three primary metrics: `sqrt(mean(roll^2))` rad;
  `mean(gx^2+gy^2)` for **unit** projected gravity; and
  `sqrt(mean((vx-vx_cmd)^2+(vy-vy_cmd)^2))` m/s using simulator body-frame truth.
  Reuse existing roll/velocity implementations. Per-trial metrics first, then
  mean and **sample** SD (`ddof=1`) across trials. These are experiment-defined
  metrics, without claimed one-to-one equivalence to other papers.
- Loss at50Hz: any ball-floor contact, or ball center wholly beyond the tray's
  outer footprint including radius (x +/-0.20m or y +/-0.15m) for >=0.1s.
  Transport loss is sticky even if the sphere later returns. Continue recording
  after sphere loss until robot failure or horizon, retaining post-loss data.
  Primary scores stop at first cargo loss. `full_window_*` are auxiliary
  diagnostics containing post-loss samples; never treat them as full-load scores.
- Robot failure: base height<0.15m or tilt>1rad; stop without automatic reset.
  All failed/lost/empty-window/invalid trials remain in the output and anomaly
  table. Censored short-rollout metrics must not be ranked as superior. Report
  transport completion alongside all three primary metrics and actual n.
- Stall: scored carried mean body vx<0.1m/s or >=2 consecutive scored seconds
  below0.1m/s. This flag never selects primary samples.

Run from `/home/hyc/unitree_rl_lab`, choosing NEW output directories each time:

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/home/hyc/miniconda3/envs/env_parkour_deploy/lib/python3.11/site-packages:${PYTHONPATH:-}"
export PYTHONEXE=/home/hyc/miniconda3/envs/env_isaaclab/bin/python

# Pure physics/metrics checks also run with env_unitree Python3.10 and MuJoCo3.6.
/home/hyc/miniconda3/envs/env_unitree/bin/python scripts/test_rolling_payload.py

# Three short trials, then another endpoint smoke at --masses8 if needed.
/home/hyc/isaac-sim/python.sh scripts/evaluate_rolling_payload.py \
  --output-dir <new-root>/smoke --masses 2 --trials 1 --duration 4

# Formal: 3 policies x4 masses x10 trials =120 new trajectories.
/home/hyc/isaac-sim/python.sh scripts/evaluate_rolling_payload.py \
  --output-dir <new-root>/formal

MPLCONFIGDIR=/tmp/rolling_payload_matplotlib \
/home/hyc/miniconda3/envs/env_unitree/bin/python scripts/report_rolling_payload.py \
  --run-root <new-root>/formal --output-dir <new-root>/summary
```

`python.sh` supplies the existing bundled Torch/RSL-RL/ONNX packages; this test
does not launch Isaac Sim, GPU simulation, DDS or real hardware. Formal runs use
MuJoCo3.11 from the named Python3.11 package path. Raw CSV, common state NPZ,
initial policy observations, preparation traces, scene copies and actual config
are saved. Missing/failed manifests invalidate a run; process exit alone does
not establish success. Output files are ignored experiment artifacts and never
committed. Source changes are committed locally, without push.

Single trained seed42 per policy; ten repetitions reflect initial-state
variation. This rolling-payload test can assess whole-policy generalization;
it cannot independently attribute gains to explicit mass estimation or establish
performance under arbitrary dynamic loads. Retain all earlier fixed-load data.

## Gentle slope and low stairs extension

Use `--terrain slope` or `--terrain stairs`; the default flat protocol above
remains available and its existing output is not overwritten. The course starts
at world x=1.5 m after a level run-up and is 6 m wide. The slope rises 5° over
3 m of horizontal travel (0.2625 m rise), ending at x=4.5 m; its top platform
continues to x=12 m. The stairs have four 0.04 m risers and 0.40 m treads; the
fourth step continues as a platform to x=12 m. Goals are x=4.9 m on the slope
and x=3.1 m on the stairs, so the robot must move beyond the final transition.
No scene geometry, mass, robot code, checkpoint or original data is changed in
the two external repositories. Evaluation-only XML copies contain the course.

The base fall threshold is local clearance above the surface (<0.15 m) or
world tilt >1 rad. The three primary scores and 2–12 s window are unchanged;
the slope's world-horizontal flatness includes its normal pitch. Additional
course completion requires robot and sphere to survive all 12 s, to visit the
course, reach the stated goal, and stay within y=±3 m. Report entrance, goal,
full course completion, carried exposure on terrain, loss and stagnation for
every trial. A low error from standing before a step is not traversal evidence.
Partial carried windows remain visible but must not be ranked over complete
traversals. Ten trials share common initial states across policies and across
the two courses, using new base seed **2026102602**; the previous flat seeds
remain distinct. All scenarios use 2/4/6/8 kg, command [0.6,0,0], 12 s and 2 s
transition with the unchanged physics, PD and observations.

Example from the repository root (choose fresh output paths):

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/home/hyc/miniconda3/envs/env_parkour_deploy/lib/python3.11/site-packages:${PYTHONPATH:-}"
export PYTHONEXE=/home/hyc/miniconda3/envs/env_isaaclab/bin/python

/home/hyc/miniconda3/envs/env_unitree/bin/python scripts/test_rolling_payload.py

/home/hyc/isaac-sim/python.sh scripts/evaluate_rolling_payload.py \
  --output-dir <new-root>/slope_formal --terrain slope --seed 2026102602
/home/hyc/isaac-sim/python.sh scripts/evaluate_rolling_payload.py \
  --output-dir <new-root>/stairs_formal --terrain stairs --seed 2026102602

MPLCONFIGDIR=/tmp/rolling_payload_matplotlib \
/home/hyc/miniconda3/envs/env_unitree/bin/python scripts/report_rolling_payload.py \
  --run-root <new-root>/slope_formal --output-dir <new-root>/slope_summary
MPLCONFIGDIR=/tmp/rolling_payload_matplotlib \
/home/hyc/miniconda3/envs/env_unitree/bin/python scripts/report_rolling_payload.py \
  --run-root <new-root>/stairs_formal --output-dir <new-root>/stairs_summary
```
