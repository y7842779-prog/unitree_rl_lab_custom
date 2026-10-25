# Three trained payload policies: MuJoCo / C++ deployment

The model exports are installed under `logs/rsl_rl/payload_comparison/` in this repository.
The original training checkout `/home/hyc/dreamwaq_go2` and all checkpoints remain read-only.
Existing verified exports were reused, including the complete PayloadEstimator actor;
no retraining or network changes were needed.

| Key after FixStand | FSM state | Model and training run | Input | Output |
| --- | --- | --- | ---: | ---: |
| `1` | AsymPPO | Asym_PPO / 2026-09-09_21-44-06 | 45 | 12 |
| `2` | DreamWaQCENet | DreamWaQ_CENet / 2026-09-11_20-51-03 | 270 | 12 |
| `3` | PayloadEstimator | PayloadEstimator / 2026-10-16_21-12-27 | 1170 | 12 |
| `v` | Velocity | The previous 2026-07-26 model, unchanged | Existing | Existing |

The original Passive, FixStand, Velocity, ParkourStudent and PIE settings are preserved.
The three new states use the existing RLBase implementation. The controller executable
and its C++ sources have not been changed or rebuilt.

## Two-terminal launch

Terminal 1:

```bash
cd ~/unitree_mujoco/simulate/build
./unitree_mujoco -i 0 -n lo -r go2 -s scene_flat.xml
```

Terminal 2:

```bash
cd ~/unitree_rl_lab/deploy/robots/go2/build
./go2_ctrl --network lo
```

Focus terminal 2. Press `f`, wait approximately 2–3 seconds for FixStand, then press
`1`, `2`, or `3` to select the desired trained model. No Enter is required.
The executable's original console hint only mentions `v`; the new numeric keys are
provided by the YAML configuration and do not require new command-line options.

Hold `w/s` for forward/backward (±0.6 m/s), `a/d` for lateral motion (±0.6 m/s),
and `q/e` for yaw (±0.8 rad/s). Release to stop the corresponding command.
The existing terminal reader handles one key at a time; it does not provide a reliable
simultaneous forward-and-yaw command. Lateral/yaw commands were increased from
±0.2 m/s and ±0.3 rad/s after isolated flat-ground trials reproduced weak responses
at those smaller commands. This changes the requested speed, not the model's
low-speed tracking ability or its training ranges. Restart `go2_ctrl` after changing
deployment commands; no compilation is needed. The original `v` policy is unchanged.

Press `f` from a new model state to return to FixStand, wait for the standing
transition, then select another model. Press `p` for Passive.
For a fresh visual comparison, stop the controller and restart the simulator between
models so that the next model does not inherit the previous robot state.

For existing obstacle terrain, replace `scene_flat.xml` with `scene_terrain.xml` in
terminal 1 and use the same scene for all methods. The original `scene.xml` contains
obstacles; `scene_flat.xml` still includes the existing modified `go2.xml`, including
its home height of 0.99 m. No XML or physical payload was changed in this migration.
These commands do not constitute newly configured 4/8 kg payload tests.

## Deployment contract

Each model directory contains `exported/policy.onnx`, `params/deploy.yaml`, and a
copy of `params/agent.yaml` for provenance. Complete training logs and checkpoints
are not copied. Generated models/configurations remain ignored by Git.

All models retain a 0.02 s policy period, kp=25, kd=0.5 and action mapping
`default_joint_position + 0.25 * raw_action`. No extra observation scaling or raw
action clipping is introduced. Additional target clamps from an older deployment
template are not applied. MuJoCo's physical joint constraints remain those in its XML.

Training joint order is hip/thigh/calf groups, each ordered FL, FR, RL, RR. Its SDK
mapping is `[3,0,9,6,4,1,10,7,5,2,11,8]`. The six current observation terms are body
angular velocity, projected gravity, command, relative joint position, joint velocity,
and previous raw action, with dimensions 3/3/3/12/12/12.
Term aliases are mapped to the existing C++ observation functions.

CENet uses current 45 plus five term-major history frames; PayloadEstimator uses
current 45 plus 25 term-major frames. Within each term, history is oldest to newest,
including the current frame; reset fills the history with its initial observation.
`use_gym_history: false` is explicit and placed after real observation terms, because
the current C++ observation manager inspects the first entry to detect input groups.

PayloadEstimator's ONNX includes CENet, velocity and latent-mean heads, the supervised
mass estimator, and the action head. It predicts normalized added mass internally;
real payload mass is not supplied to the deployed actor.

## Validation and its limits

Migration checks include saved-configuration values, all ONNX parameter initializers
against the corresponding final checkpoint, strict PyTorch checkpoint loading, ONNX
graph validity, 40 frames of exact C++ history/observation agreement, and C++ ONNX
actions against the original PyTorch actor.

| Model | Maximum absolute raw action difference |
| --- | ---: |
| Asym_PPO | 1.1921e-6 |
| DreamWaQ_CENet | 8.3447e-7 |
| PayloadEstimator | 8.7917e-7 |

The separate checker links the existing C++ ONNX Runtime; it does not initialize DDS.
No simulator, closed-loop rollout or parameter update was run in this migration.
These checks validate export and interface compatibility, not successful locomotion.
Subsequent isolated headless MuJoCo trials used the original checkpoint actors,
the local `scene_flat.xml`, a 0.002 s physics step and a 0.02 s policy period.
Each trial used the default joint pose at z=0.4 m, 1 s PD settling, 1 s of zero
policy command, then 7 s of constant command. Final-3-second mean lateral speeds
at a 0.6 m/s request were 0.419/0.420/0.429 m/s for Asym/CENet/Payload; mean yaw
rates at a 0.8 rad/s request were 0.475/0.578/0.506 rad/s. These diagnostics motivate
the new keyboard magnitudes; they do not establish tracking performance in the
interactive DDS run. Small-command tracking remains a separate evaluation case.

Lowstate CSVs are named by the selected model under `results/payload_comparison/`.
The existing logger is enabled while the model state is active. Files may be created
for inactive states at controller initialization; do not count empty files as trials.
It records joints, IMU and joystick inputs, but **not the actual keyboard command or
full body linear velocity**. Joystick-command zeros do not mean the keyboard sent zero.
These logs cannot directly support keyboard velocity-tracking RMSE. A quantitative
benchmark still requires actual command and body-velocity recording plus matched
terrain, physical load, initial state and time; the visual launch is not that benchmark.

See `logs/rsl_rl/payload_comparison/validation.csv`, `migration_manifest.json` and
the per-model `cpp_observations.txt` / `cpp_actions.txt` for reproducible validation.
The original controller configuration is backed up as
`logs/rsl_rl/payload_comparison/config_before_migration.yaml`.

## Reproduce migration

```bash
cd ~/unitree_rl_lab
PYTHONDONTWRITEBYTECODE=1 \
PYTHONEXE=/home/hyc/miniconda3/envs/env_isaaclab/bin/python \
/home/hyc/isaac-sim/python.sh scripts/migrate_payload_comparison.py \
  --destination-root logs/rsl_rl/payload_comparison_reproduce_new
```

An existing destination is rejected. The script reads the preserved verified bundle
at `dreamwaq_go2/logs/mujoco_preview/2026-10-25_v2` and original checkpoints. It does
not change the controller YAML; if another destination is used, model-selection paths
must be configured separately. The existing Python launcher only supplies dependencies,
and no Isaac Sim application or new dependency installation is involved.

New bundles default to lateral ±0.6 m/s and yaw ±0.8 rad/s. To reproduce the
original smaller keyboard commands, append
`--keyboard-lateral-speed 0.2 --keyboard-yaw-speed 0.3` to the migration command.
Both options must be finite, positive and within the saved training ranges.
Existing installed bundles still require their `params/deploy.yaml` values to be
updated explicitly; the migration script continues to reject existing destinations.

## Git scope

Only `/home/hyc/unitree_rl_lab` on branch `main` changes: the Go2 YAML adds model
selection, `scripts/migrate_payload_comparison.py` supplies repeatable migration and
CPU checks, and this document describes use and evidence limits. Models, generated
configuration, validation files and logs are not committed.
Baseline commit: `8b9d397` (`chore: mark baseline before three-policy payload deployment migration`).
The implementation commit and final states of all repositories are recorded in
`logs/rsl_rl/payload_comparison/delivery_manifest.json` at delivery. Nothing is pushed.
The training repository's pre-existing cache/diagram changes and the MuJoCo repository's
pre-existing source/scene changes are preserved.
