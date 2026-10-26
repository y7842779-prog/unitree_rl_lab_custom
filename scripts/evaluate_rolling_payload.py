"""Paired deterministic inference with a freely moving sphere in the Oracle tray.

Original XML and checkpoints are read only. All scenes and raw trial files are
written to a new results directory; no DDS, training or policy interventions.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import subprocess
import traceback

import mujoco
import numpy as np
import torch
import yaml

from migrate_payload_comparison import RUNS, DataLoader, actor_from_checkpoint, validate_deploy
from rolling_payload_support import (MODELS, CargoMonitor, ball_contacts, build_scene,
    indices, initial_samples, metrics, observation, sha, state_values, write_csv, write_json)

ROOT = Path(__file__).resolve().parents[1]
SIGNATURE = mujoco.mjtState.mjSTATE_INTEGRATION


def state_vector(model, data):
    result = np.empty(mujoco.mj_stateSize(model,SIGNATURE))
    mujoco.mj_getState(model,data,result,SIGNATURE)
    return result


def prepare(model, idx, cfg, sample, directory, settle_s):
    """Prepare once per mass/trial, under common stand PD, before any actor runs."""
    data = mujoco.MjData(model)
    default = np.asarray(cfg["default_joint_pos"])
    data.qpos[:2] = sample["xy"]
    data.qpos[2] = .4
    mujoco.mju_euler2Quat(data.qpos[3:7],np.asarray(sample["rpy"]),"xyz")
    data.qpos[idx["qidx"]] = default+sample["joint_noise"]
    data.qvel[idx["vidx"]] = sample["joint_velocity"]
    mujoco.mj_forward(model,data)
    rotation = data.xmat[idx["base"]].reshape(3,3)
    adr = idx["ball_qidx"]
    data.qpos[adr:adr+3] = data.xpos[idx["base"]]+rotation@np.array([0.,0.,.099])
    data.qpos[adr+3:adr+7] = [1.,0.,0.,0.]
    data.qvel[idx["ball_vidx"]:idx["ball_vidx"]+6] = 0.
    mujoco.mj_forward(model,data)
    original = state_vector(model,data).copy()
    contacts = ball_contacts(model,data,idx)
    assert contacts["ball_min_contact_gap_m"] == "" or contacts["ball_min_contact_gap_m"] >= -1e-8
    decimation = round(.02/model.opt.timestep)
    rows, monitor = [], CargoMonitor()
    for step in range(round(settle_s/.02)):
        for _ in range(decimation):
            data.ctrl[idx["actuator"]] = 25.*(default-data.qpos[idx["qidx"]])-.5*data.qvel[idx["vidx"]]
            mujoco.mj_step(model,data)
        mujoco.mj_forward(model,data)
        values = state_values(model,data,idx)
        lost = monitor.update(values,(step+1)*.02)
        rows.append(dict(step=step,time_s=(step+1)*.02,cargo_lost=lost,**values))
    write_csv(directory/f"trial_{sample['trial']:02d}_settle.csv",rows)
    last = rows[-1]
    if (monitor.first_loss_s is not None or last["height"] < .15 or last["tilt_cosine"] < math.cos(1.)
            or sum(r["ball_robot_contacts"]>0 for r in rows[-10:]) < 5):
        raise RuntimeError(f"Common preparation failed before any policy: mass={last['true_mass_kg']}, trial={sample['trial']}, last={last}")
    state = state_vector(model,data)
    path = directory/f"trial_{sample['trial']:02d}_state.npz"
    np.savez_compressed(path,state=state,original_state=original,qpos=data.qpos.copy(),qvel=data.qvel.copy())
    return state, dict(trial=sample["trial"],evaluation_seed=sample["evaluation_seed"],
        path=str(path),sha256=sha(path),state_signature=int(SIGNATURE),
        initial_ball_penetration_m=min(0.,contacts["ball_min_contact_gap_m"]) if contacts["ball_min_contact_gap_m"]!="" else 0.,
        final_settle_state=last)


def load_policies(args):
    policies, configs, metadata, protected = {}, {}, {}, {}
    for label,(date,dim,_) in RUNS.items():
        run = args.source_root/"logs/rsl_rl/payload_comparison"/label/date
        deploy = args.policy_root/label/date/"params/deploy.yaml"
        for path in (run/"model_4999.pt",run/"params/env.yaml",run/"params/agent.yaml",deploy):
            protected[str(path)] = sha(path)
        cfg = yaml.safe_load(deploy.read_text())
        env = yaml.load((run/"params/env.yaml").read_text(),Loader=DataLoader)
        contract = dict(cfg,observations={k:v for k,v in cfg["observations"].items() if k!="use_gym_history"})
        validate_deploy(contract,env,dim)
        agent = yaml.safe_load((run/"params/agent.yaml").read_text())
        assert agent["clip_actions"] is None and not agent["empirical_normalization"]
        saved = torch.load(run/"model_4999.pt",map_location="cpu",weights_only=True)
        assert saved["iter"] == 4999
        with contextlib.redirect_stdout(io.StringIO()):
            actor = actor_from_checkpoint(args.source_root,run,label,saved["model_state_dict"])
        with torch.inference_mode():
            probe = torch.zeros(1,dim)
            assert torch.equal(actor(probe),actor(probe))
        policies[label],configs[label] = actor,cfg
        metadata[label] = dict(checkpoint=str(run/"model_4999.pt"),checkpoint_sha256=protected[str(run/"model_4999.pt")],
            agent_config=str(run/"params/agent.yaml"),env_config=str(run/"params/env.yaml"),
            actor_observation_dim=dim,iteration=4999,strict_load=True,training_seed=agent["seed"],
            policy_config=agent["policy"],action_scale=.25,kp=25.,kd=.5,control_dt=.02,
            joint_order=__import__("rolling_payload_support").JOINTS)
    for cfg in configs.values():
        for key in ("default_joint_pos","stiffness","damping","step_dt","actions","joint_ids_map"):
            assert cfg[key] == configs["Asym_PPO"][key]
    return policies,configs,metadata,protected


def rollout(model, idx, initial, sample, label, actor, cfg, args, directory):
    data = mujoco.MjData(model)
    mujoco.mj_setState(model,data,initial,SIGNATURE)
    np.testing.assert_array_equal(state_vector(model,data),initial)
    mujoco.mj_forward(model,data)
    default = np.asarray(cfg["default_joint_pos"],dtype=np.float32)
    start = data.qpos[:2].copy()
    dt = cfg["step_dt"]
    decimation = round(dt/model.opt.timestep)
    assert math.isclose(decimation*model.opt.timestep,dt,abs_tol=1e-12)
    previous = np.zeros(12,dtype=np.float32)
    command = np.asarray([.6,0.,0.],dtype=np.float32)
    history, rows, monitor = [], [], CargoMonitor(dt)
    mass_snapshot, inertia_snapshot = model.body_mass.copy(), model.body_inertia.copy()
    with torch.inference_mode():
        for step in range(round(args.duration/dt)):
            rotation = data.xmat[idx["base"]].reshape(3,3)
            frame = np.concatenate((data.sensor("imu_gyro").data,rotation.T@[0.,0.,-1.],command,
                data.qpos[idx["qidx"]]-default,data.qvel[idx["vidx"]],previous)).astype(np.float32)
            obs,history = observation(frame,history,RUNS[label][1]//45-1)
            assert obs.shape == (RUNS[label][1],)
            if step == 0:
                np.savez_compressed(directory/f"trial_{sample['trial']:02d}_initial_observation.npz",
                                    current=frame,observation=obs)
            tensor = torch.from_numpy(obs).unsqueeze(0)
            previous = actor(tensor)[0].numpy()
            if not np.isfinite(previous).all():
                raise FloatingPointError(f"Nonfinite actor action: {label}")
            predicted = float(actor.predict_payload(tensor).item()*actor.payload_scale_kg) if label=="PayloadEstimator" else ""
            target = default+.25*previous
            for _ in range(decimation):
                data.ctrl[idx["actuator"]] = 25.*(target-data.qpos[idx["qidx"]])-.5*data.qvel[idx["vidx"]]
                mujoco.mj_step(model,data)
            mujoco.mj_forward(model,data)
            np.testing.assert_array_equal(model.body_mass,mass_snapshot)
            np.testing.assert_array_equal(model.body_inertia,inertia_snapshot)
            values = state_values(model,data,idx)
            lost = monitor.update(values,(step+1)*dt)
            failed = values["height"] < .15 or values["tilt_cosine"] < math.cos(1.)
            row = dict(model=label,mass_kg=float(model.body_mass[idx["ball"]]),trial=sample["trial"],
                evaluation_seed=sample["evaluation_seed"],training_seed=42,step=step,
                time_before_s=step*dt,time_after_s=(step+1)*dt,scored=int(step*dt>=args.warmup-1e-9),
                cmd_vx=float(command[0]),cmd_vy=0.,cmd_wz=0.,predicted_mass_kg=predicted,
                x_progress=float(data.qpos[0]-start[0]),y_progress=float(data.qpos[1]-start[1]),
                cargo_lost=lost,terminated=int(failed),timeout=0,**values,
                **{f"action_{i}":float(a) for i,a in enumerate(previous)})
            rows.append(row)
            if failed:
                break
    path = directory/f"trial_{sample['trial']:02d}.csv"
    write_csv(path,rows)
    episode = dict(model=label,mass_kg=float(model.body_mass[idx["ball"]]),trial=sample["trial"],
        evaluation_seed=sample["evaluation_seed"],training_seed=42,**metrics(rows,args.duration,args.warmup))
    write_json(directory/f"trial_{sample['trial']:02d}_metrics.json",episode)
    return episode


def main(args):
    torch.set_num_threads(1)
    policies,configs,metadata,protected = load_policies(args)
    source_robot = args.scene.parent/"go2.xml"
    for path in (args.scene,source_robot):
        protected[str(path)] = sha(path)
    samples = initial_samples(args.seed,args.trials)
    assert not ({s["evaluation_seed"] for s in samples} &
        {s["evaluation_seed"] for prior in (20261025,20261026) for s in initial_samples(prior,10)})
    write_json(args.output_dir/"initial_samples.json",samples)
    protocol = dict(status="specified before rollouts",args=vars(args),checkpoints=metadata,
        payload_model="Free sphere with unchanged radius/contacts; base nominal mass; sphere inertia compiled from geom mass",
        comparison="MuJoCo rolling-payload generalization; independent of Isaac Lab/base-mass results",
        initial_state="One common 2 s stand-PD preparation per mass and trial, full integration state restored for all three actors",
        observation="Original 45/270/1170 inputs, term-major history, deterministic actor/latent means; no true ball state or mass in actor",
        command=[.6,0.,0.],physics_dt=.002,control_dt=.02,
        main_metrics="Roll RMS rad; mean(gx^2+gy^2) for unit body gravity; planar body velocity vector RMSE m/s",
        score_window="Intervals beginning at >=2s, post-step truth (2,12]s; 500 samples when complete",
        loss_definition="At 50Hz: ball-floor contact OR sphere center beyond tray outer x +/-0.16 plus radius0.04 or y +/-0.11 plus radius0.04 for >=0.1s",
        censoring="Main metrics end at first cargo loss or robot failure; all trials retained and counts shown. Full-window auxiliary metrics retain post-loss data and are NOT full-load scores.",
        robot_failure="Height <0.15m OR tilt >1rad; no automatic resets",
        motion_check="Scored carried mean body vx<0.1m/s OR consecutive >=2s body vx<0.1m/s",
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        git_branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True),
        mujoco_version=mujoco.__version__,torch_version=torch.__version__,numpy_version=np.__version__)
    write_json(args.output_dir/"protocol.json",protocol)
    episodes,preparations = [],[]
    for mass in args.masses:
        model,physics = build_scene(args.scene,args.output_dir/f"scene_{mass:g}kg",mass)
        idx = indices(model)
        initial_dir = args.output_dir/f"initial_{mass:g}kg"
        initial_dir.mkdir()
        case_dirs = {}
        for label in MODELS:
            path = args.output_dir/f"mass_{mass:g}kg"/label
            path.mkdir(parents=True)
            case_dirs[label] = path
        for sample in samples:
            initial,info = prepare(model,idx,configs["Asym_PPO"],sample,initial_dir,args.settle)
            preparations.append(dict(mass_kg=mass,**info))
            for label in MODELS:
                episode = rollout(model,idx,initial,sample,label,policies[label],configs[label],args,case_dirs[label])
                episodes.append(episode)
                print("TRIAL_COMPLETE",mass,label,sample["trial"],"transport",episode["transport_completed"],
                      "loss",episode["cargo_lost"],"robot_fail",episode["failed"],flush=True)
        for label,path in case_dirs.items():
            selected = [e for e in episodes if e["mass_kg"]==mass and e["model"]==label]
            write_csv(path/"episode_metrics.csv",selected)
            write_json(path/"manifest.json",dict(status="complete",model=label,mass_kg=mass,
                trials=len(selected),checkpoint=metadata[label],physics=physics,
                initial_state_sha256=[p["sha256"] for p in preparations if p["mass_kg"]==mass],
                exact_integration_state_pairing=True,
                transport_completed=sum(e["transport_completed"] for e in selected)))
    write_csv(args.output_dir/"episode_metrics.csv",episodes)
    write_json(args.output_dir/"preparation_checks.json",preparations)
    assert all(sha(p)==value for p,value in protected.items())
    write_json(args.output_dir/"manifest.json",dict(status="complete",trajectories=len(episodes),
        trials_per_condition=args.trials,masses=args.masses,protected_sha256=protected,checkpoints=metadata,
        transport_completed=sum(e["transport_completed"] for e in episodes),
        robot_completed=sum(e["robot_completed"] for e in episodes),
        cargo_losses=sum(e["cargo_lost"] for e in episodes)))
    print("ROLLING_COMPLETE",args.output_dir,"trajectories",len(episodes),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,default=Path("/home/hyc/dreamwaq_go2"))
    parser.add_argument("--policy-root",type=Path,default=ROOT/"logs/rsl_rl/payload_comparison")
    parser.add_argument("--scene",type=Path,default=Path("/home/hyc/unitree_mujoco/unitree_robots/go2_Oracle/scene.xml"))
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--masses",type=float,nargs="+",default=[2.,4.,6.,8.])
    parser.add_argument("--trials",type=int,default=10)
    parser.add_argument("--seed",type=int,default=2026102601)
    parser.add_argument("--duration",type=float,default=12.)
    parser.add_argument("--warmup",type=float,default=2.)
    parser.add_argument("--settle",type=float,default=2.)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        parser.error("Use a new output directory; existing experiments are never overwritten")
    if (args.trials<1 or not 0<=args.warmup<args.duration or args.settle<=0 or
            len(set(args.masses))!=len(args.masses) or not all(math.isfinite(m) and 0<m<=8 for m in args.masses)):
        parser.error("Invalid trials, duration, warmup, settle time or masses")
    args.output_dir.mkdir(parents=True)
    try:
        main(args)
    except BaseException:
        write_json(args.output_dir/"error.json",dict(status="failed",traceback=traceback.format_exc()))
        raise
