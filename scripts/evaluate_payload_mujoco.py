"""Fresh, paired flat-ground payload trials; synchronous CPU inference, no DDS."""

from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import io
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from migrate_payload_comparison import RUNS, JOINTS, DataLoader, actor_from_checkpoint, validate_deploy
from payload_metrics import sha256, trial_metrics, write_summaries, yaw_from_quaternion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/home/hyc/dreamwaq_go2"))
    parser.add_argument("--policy-root", type=Path, default=Path(__file__).resolve().parents[1] / "logs/rsl_rl/payload_comparison")
    parser.add_argument("--scene", type=Path, default=Path("/home/hyc/unitree_mujoco/unitree_robots/go2/scene_flat.xml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--warmup", type=float, default=2.0)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output exists; choose a new directory")
    if args.trials < 1 or not 0 <= args.warmup < args.duration:
        parser.error("Require trials > 0 and 0 <= warmup < duration")
    torch.set_num_threads(1)
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    protected = {str(p): sha256(p) for p in (args.scene, args.scene.parent / "go2.xml")}
    rng = np.random.default_rng(args.seed)
    initial_states = []
    for trial in range(args.trials):
        rpy = rng.uniform([-0.01, -0.01, -0.03], [0.01, 0.01, 0.03])
        quat = np.empty(4)
        mujoco.mju_euler2Quat(quat, rpy, "xyz")
        initial_states.append(dict(trial=trial, xy=rng.uniform(-0.01, 0.01, 2).tolist(),
                                   quaternion_wxyz=quat.tolist(), joint_noise=rng.uniform(-0.005, 0.005, 12).tolist()))
    episodes, physical_models, checkpoint_info = [], {}, {}
    for mass in (0.0, 4.0, 8.0):
        model = mujoco.MjModel.from_xml_path(str(args.scene))
        base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        nominal = float(model.body_mass[base])
        model.body_mass[base] = nominal + mass
        model.body_inertia[base] *= (nominal + mass) / nominal
        physical_models[mass] = model
    with gzip.open(output / "raw_trajectories.csv.gz", "wt", encoding="utf-8", newline="") as raw_handle:
        writer = None
        for label, (date, dim, _) in RUNS.items():
            run = args.source_root / "logs/rsl_rl/payload_comparison" / label / date
            config_path = args.policy_root / label / date / "params/deploy.yaml"
            for path in (run / "model_4999.pt", run / "params/agent.yaml", run / "params/env.yaml", config_path):
                protected[str(path)] = sha256(path)
            cfg = yaml.safe_load(config_path.read_text())
            contract = dict(cfg, observations={k: v for k, v in cfg["observations"].items() if k != "use_gym_history"})
            env_cfg = yaml.load((run / "params/env.yaml").read_text(), Loader=DataLoader)
            validate_deploy(contract, env_cfg, dim)
            checkpoint = torch.load(run / "model_4999.pt", map_location="cpu", weights_only=True)
            with contextlib.redirect_stdout(io.StringIO()):
                actor = actor_from_checkpoint(args.source_root, run, label, checkpoint["model_state_dict"])
            checkpoint_info[label] = dict(path=str(run / "model_4999.pt"), sha256=protected[str(run / "model_4999.pt")],
                                          training_seed=42, input_dim=dim)
            default = np.asarray(cfg["default_joint_pos"], dtype=np.float32)
            mapping = np.asarray(cfg["joint_ids_map"], dtype=int)
            kp, kd = np.asarray(cfg["stiffness"])[mapping], np.asarray(cfg["damping"])[mapping]
            scale = np.asarray(cfg["actions"]["JointPositionAction"]["scale"])
            history_length = dim // 45 - 1
            for mass, model in physical_models.items():
                base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
                joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINTS]
                qidx, vidx = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
                actuator_idx = np.asarray([np.flatnonzero(model.actuator_trnid[:, 0] == j).item() for j in joints])
                decimation = round(cfg["step_dt"] / model.opt.timestep)
                if not math.isclose(decimation * model.opt.timestep, cfg["step_dt"], abs_tol=1e-9):
                    raise ValueError("Policy period is not an integer physics step count")
                for yaw_command in (0.0, 0.15):
                    for initial in initial_states:
                        data = mujoco.MjData(model)
                        mujoco.mj_setConst(model, data)
                        data.qpos[:7] = [*initial["xy"], 0.4, *initial["quaternion_wxyz"]]
                        data.qpos[qidx] = default + np.asarray(initial["joint_noise"])
                        mujoco.mj_forward(model, data)
                        for _ in range(round(1 / model.opt.timestep)):
                            data.ctrl[actuator_idx] = kp * (default - data.qpos[qidx]) - kd * data.qvel[vidx]
                            mujoco.mj_step(model, data)
                        mujoco.mj_forward(model, data)
                        start = data.qpos[:2].copy()
                        start_yaw = yaw_from_quaternion(data.qpos[3:7])
                        command = np.asarray([0.6, 0, yaw_command], dtype=np.float32)
                        previous = np.zeros(12, dtype=np.float32)
                        history, rows = [], []
                        with torch.inference_mode():
                            for step in range(round(args.duration / cfg["step_dt"])):
                                rotation = data.xmat[base].reshape(3, 3)
                                frame = np.concatenate((data.sensor("imu_gyro").data,
                                    rotation.T @ [0, 0, -1], command, data.qpos[qidx] - default,
                                    data.qvel[vidx], previous)).astype(np.float32)
                                if history_length:
                                    history = history[1:] + [frame] if history else [frame] * history_length
                                    history_array = np.stack(history)
                                    offset, terms = 0, []
                                    for width in (3, 3, 3, 12, 12, 12):
                                        terms.append(history_array[:, offset:offset + width].reshape(-1))
                                        offset += width
                                    observation = np.concatenate([frame] + terms)
                                else:
                                    observation = frame
                                if len(observation) != dim:
                                    raise ValueError("Observation dimension mismatch")
                                tensor = torch.from_numpy(observation).unsqueeze(0)
                                previous = actor(tensor)[0].numpy()
                                if not np.isfinite(previous).all():
                                    raise ValueError("Nonfinite policy action")
                                estimated_mass = float(actor.predict_payload(tensor).item() * actor.payload_scale_kg) if label == "PayloadEstimator" else ""
                                target = default + scale * previous
                                for _ in range(decimation):
                                    data.ctrl[actuator_idx] = kp * (target - data.qpos[qidx]) - kd * data.qvel[vidx]
                                    mujoco.mj_step(model, data)
                                mujoco.mj_forward(model, data)
                                rotation = data.xmat[base].reshape(3, 3)
                                vx, vy = (rotation.T @ data.qvel[:3])[:2]
                                wz = float(data.sensor("imu_gyro").data[2])
                                w, x, y, z = data.qpos[3:7]
                                roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
                                pitch = math.asin(np.clip(2 * (w * y - z * x), -1, 1))
                                failed = data.qpos[2] < 0.15 or rotation[2, 2] < math.cos(1.0)
                                row = dict(model=label, terrain="flat", true_mass_kg=mass,
                                           trial=initial["trial"], evaluation_seed=args.seed, training_seed=42,
                                           step=step, time_before_s=step * cfg["step_dt"], time_after_s=(step + 1) * cfg["step_dt"],
                                           scored=int(step * cfg["step_dt"] >= args.warmup - 1e-8),
                                           cmd_vx=float(command[0]), cmd_vy=0.0, cmd_wz=float(command[2]),
                                           vx=float(vx), vy=float(vy), wz=wz, roll=roll, pitch=pitch,
                                           x_progress=float(data.qpos[0] - start[0]), y_progress=float(data.qpos[1] - start[1]),
                                           height=float(data.qpos[2]), predicted_mass_kg=estimated_mass,
                                           terminated=int(failed), timeout=0,
                                           **{f"action_{j}": float(a) for j, a in enumerate(previous)})
                                if writer is None:
                                    writer = csv.DictWriter(raw_handle, fieldnames=list(row))
                                    writer.writeheader()
                                writer.writerow(row)
                                rows.append(row)
                                if failed:
                                    break
                        episodes.append(dict(engine="MuJoCo_new", condition=f"flat_mass{mass:g}_wz{yaw_command:g}",
                                             model=label, terrain="flat", mass_kg=mass, cmd_wz=yaw_command,
                                             trial=initial["trial"], evaluation_seed=args.seed, training_seed=42,
                                             **trial_metrics(rows, args.duration, start_yaw)))
                    print(label, "mass", mass, "yaw", yaw_command, "trials", args.trials, flush=True)
    write_summaries(output, episodes)
    if not all(sha256(p) == digest for p, digest in protected.items()):
        raise ValueError("Input files changed during evaluation")
    (output / "manifest.json").write_text(json.dumps(dict(
        new_rollouts=len(episodes), reused_rollouts=0, checkpoints=checkpoint_info,
        protocol=dict(scene=str(args.scene), initial_height_m=0.4, pd_settle_s=1,
                      duration_s=args.duration, warmup_s=args.warmup, initial_states=initial_states,
                      masses_kg=[0, 4, 8], commands=[[0.6, 0, 0], [0.6, 0, 0.15]],
                      trials_per_condition=args.trials, seed=args.seed, physics_dt=float(model.opt.timestep), policy_dt=0.02,
                      added_mass_model="Increase base mass and scale its inertia in proportion; COM fixed; no attached geometry",
                      failure="Base height <0.15 m or tilt >1 rad; no task-completion threshold applied"),
        mujoco_version=mujoco.__version__, torch_version=torch.__version__, protected_sha256=protected,
        limitations=["Independent synchronous flat-ground validation, without DDS, terminal keyboard, or real hardware.",
                     "No slope/stair or dynamic load transitions in this new dataset.",
                     "Five paired initial poses per condition by default; one trained seed per method.",
                     "Do not pool these measurements with the Isaac Lab dataset."]
    ), indent=2))
    print("MUJOCO_COMPLETE", output, "trials", len(episodes), flush=True)


if __name__ == "__main__":
    main()
