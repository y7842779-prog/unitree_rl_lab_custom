# Transfer files for the payload comparison paper

The paper artifacts live in three Git repositories. Clone each repository at
the branch used by the experiment, and run `git lfs pull` after cloning.

| Repository | Required contents |
| --- | --- |
| `dreamwaq_go2` | Isaac Lab task/training/evaluation source, three final `model_4999.pt` checkpoints and their saved `env.yaml`/`agent.yaml`, plus `reproduction_artifacts/isaac_three_metrics_2026-10-26.zip` |
| `unitree_rl_lab` | MuJoCo rolling-payload evaluation source, three deployment YAML/ONNX exports in `logs/rsl_rl/payload_comparison`, plus the flat and gentle-terrain result ZIP files under `results` |
| `go2_unitree_mujoco` | `unitree_robots/go2_Oracle/scene.xml`, `go2.xml`, and all 16 `assets/*.obj` meshes |

For the MuJoCo evaluation, provide the new machine's actual paths with
`--source-root`, `--policy-root`, and `--scene`; the defaults in the script use
the original workstation paths. Rebuild evaluation scenes from the Oracle
source XML. Generated `robot.xml` files in the result archives contain the old
machine's absolute mesh directory. The standalone unit test also has a fixed
scene path at the top of `scripts/test_rolling_payload.py`.

The model files alone cannot run Isaac Lab training or Isaac Lab evaluation.
Install Isaac Sim, Isaac Lab and RSL-RL separately, with the official Go2 USD
asset available. The original experiment used the local Isaac Lab 2.2.0
checkout; framework changes that were not committed at training time require
separate verification before claiming an exact training reproduction.

The two ZIP files under `results` contain raw MuJoCo traces, scene copies,
summary CSVs, PNG/PDF plots, input checksums and validation results. The Isaac
three-metric ZIP contains the corresponding Isaac Lab flat-ground evaluation.
These result archives are data; the checkpoints and Oracle meshes remain in
their respective repositories.
