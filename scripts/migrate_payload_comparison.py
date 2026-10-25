"""Copy verified final policy exports into this repository and check CPU inference.

The training checkout is read-only. This script never starts DDS or a simulator,
changes the controller selection, or overwrites an existing destination.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import torch
import yaml

RUNS = {
    "Asym_PPO": ("2026-09-09_21-44-06", 45, 238),
    "DreamWaQ_CENet": ("2026-09-11_20-51-03", 270, 238),
    "PayloadEstimator": ("2026-10-16_21-12-27", 1170, 239),
}
TERMS = ("base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions")
CPP_TERMS = ("base_ang_vel", "projected_gravity", "keyboard_velocity_commands", "joint_pos_rel", "joint_vel_rel", "last_action")
WIDTHS = (3, 3, 3, 12, 12, 12)
JOINTS = [f"{leg}_{kind}_joint" for kind in ("hip", "thigh", "calf") for leg in ("FL", "FR", "RL", "RR")]
SDK = [f"{leg}_{kind}_joint" for leg in ("FR", "FL", "RR", "RL") for kind in ("hip", "thigh", "calf")]


class DataLoader(yaml.SafeLoader):
    pass


def tagged(loader, tag, node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return loader.construct_scalar(node)


DataLoader.add_multi_constructor("tag:yaml.org,2002:python/", tagged)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def actor_from_checkpoint(source, run, model, state):
    agent = yaml.safe_load((run/"params/agent.yaml").read_text())
    assert not agent["empirical_normalization"]
    cfg = dict(agent["policy"])
    class_name = cfg.pop("class_name")
    if model == "Asym_PPO":
        from rsl_rl.modules import ActorCritic
        assert class_name == "ActorCritic"
        network = ActorCritic(45, 238, 12, **cfg)
    else:
        name = "_payload_migration_agents"
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(source/"source/dreamwaq_go2/dreamwaq_go2/tasks/manager_based/dreamwaq_go2/agents")]
            sys.modules[name] = package
        module = "payload_actor_critic" if model == "PayloadEstimator" else "dreamwaq_actor_critic"
        cls = getattr(importlib.import_module(name+"."+module), class_name)
        network = cls(RUNS[model][1], RUNS[model][2], 12, **cfg)
    torch.nn.Module.load_state_dict(network, state, strict=True)
    return network.eval().actor


def validate_deploy(cfg, env, dim):
    assert cfg["step_dt"] == env["sim"]["dt"]*env["decimation"] == .02
    assert cfg["joint_ids_map"] == [SDK.index(j) for j in JOINTS]
    defaults = []
    for joint in JOINTS:
        values = [v for expr,v in env["scene"]["robot"]["init_state"]["joint_pos"].items() if re.fullmatch(expr,joint)]
        assert len(values) == 1
        defaults.append(values[0])
    assert cfg["default_joint_pos"] == defaults
    actuator = env["scene"]["robot"]["actuators"]["base_legs"]
    assert cfg["stiffness"] == [actuator["stiffness"]]*12 == [25.]*12
    assert cfg["damping"] == [actuator["damping"]]*12 == [.5]*12
    action = cfg["actions"]["JointPositionAction"]
    assert action["scale"] == [env["actions"]["joint_pos"]["scale"]]*12 == [.25]*12
    assert action["offset"] == defaults and action["clip"] is None
    obs = cfg["observations"]
    names = list(CPP_TERMS)
    if dim > 45:
        names += ["history_"+n for n in CPP_TERMS]
    assert list(obs) == names
    for index,name in enumerate(names):
        source_name = ("history_" if index>=6 else "")+TERMS[index%6]
        original = env["observations"]["policy"][source_name]
        assert obs[name]["history_length"] == max(1,original["history_length"])
        assert obs[name]["scale"] is None and original["scale"] is None
        assert obs[name]["clip"] is None and original["clip"] is None
    assert sum(WIDTHS[i%6]*obs[n]["history_length"] for i,n in enumerate(names)) == dim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/home/hyc/dreamwaq_go2"))
    parser.add_argument("--preview-root", type=Path)
    parser.add_argument("--destination-root", type=Path)
    parser.add_argument("--keyboard-lateral-speed", type=float, default=0.6,
                        help="Keyboard lateral command magnitude in m/s (default: 0.6)")
    parser.add_argument("--keyboard-yaw-speed", type=float, default=0.8,
                        help="Keyboard yaw command magnitude in rad/s (default: 0.8)")
    args = parser.parse_args()
    if not all(np.isfinite(v) and v > 0 for v in (args.keyboard_lateral_speed, args.keyboard_yaw_speed)):
        parser.error("Keyboard speeds must be finite and positive")
    repo = Path(__file__).resolve().parents[1]
    source = args.source_root.resolve()
    preview = (args.preview_root or source/"logs/mujoco_preview/2026-10-25_v2").resolve()
    dest = (args.destination_root or repo/"logs/rsl_rl/payload_comparison").resolve()
    if dest.exists():
        parser.error("Destination exists; use a new directory")
    meta = json.loads((preview/"manifest.json").read_text())
    protected = {}
    for model,(date,_,_) in RUNS.items():
        run = source/"logs/rsl_rl/payload_comparison"/model/date
        for path in (run/"model_4999.pt",run/"params/agent.yaml",run/"params/env.yaml",preview/model/"exported/policy.onnx",preview/model/"params/deploy.yaml"):
            protected[str(path)] = sha(path)
        assert protected[str(run/"model_4999.pt")] == meta["models"][model]["checkpoint_sha256"]
    dest.mkdir(parents=True)
    validation = dest/"validation"
    validation.mkdir()
    shutil.copy2(preview/"check_observations.cpp",validation/"check_observations.cpp")
    shutil.copy2(preview/"synthetic_frames.txt",validation/"synthetic_frames.txt")
    ort = repo/"deploy/thirdparty/onnxruntime-linux-x64-1.22.0"
    checker = validation/"check_observations"
    subprocess.run(["g++","-std=c++17","-O2","-I"+str(repo/"deploy/include"),"-I"+str(ort/"include"),
        str(validation/"check_observations.cpp"),"-lyaml-cpp",str(ort/"lib/libonnxruntime.so.1"),
        "-Wl,-rpath,"+str(ort/"lib"),"-o",str(checker)],check=True)
    rows = []
    for model,(date,dim,_) in RUNS.items():
        run = source/"logs/rsl_rl/payload_comparison"/model/date
        target = dest/model/date
        (target/"exported").mkdir(parents=True)
        (target/"params").mkdir()
        cfg = yaml.safe_load((preview/model/"params/deploy.yaml").read_text())
        env = yaml.load((run/"params/env.yaml").read_text(),Loader=DataLoader)
        validate_deploy(cfg,env,dim)
        # Explicit deployment commands; keep training and forward-command ranges intact.
        for axis,speed in (("lin_vel_y",args.keyboard_lateral_speed), ("ang_vel_z",args.keyboard_yaw_speed)):
            lower,upper = env["commands"]["base_velocity"]["ranges"][axis]
            if not lower <= -speed <= speed <= upper:
                raise ValueError(f"{model}: keyboard {axis} +/-{speed} exceeds training range [{lower}, {upper}]")
            cfg["commands"]["base_velocity"]["ranges"][axis] = [-speed,speed]
        # Keep the first observation a real term: the C++ manager inspects it.
        cfg["observations"]["use_gym_history"] = False
        (target/"params/deploy.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False))
        shutil.copy2(run/"params/agent.yaml",target/"params/agent.yaml")
        policy = target/"exported/policy.onnx"
        shutil.copy2(preview/model/"exported/policy.onnx",policy)
        graph = onnx.load(str(policy))
        onnx.checker.check_model(graph)
        assert graph.graph.input[0].name == "obs"
        assert graph.graph.input[0].type.tensor_type.shape.dim[1].dim_value == dim
        saved = torch.load(run/"model_4999.pt",map_location="cpu",weights_only=True)
        weights = saved["model_state_dict"]
        for initializer in graph.graph.initializer:
            np.testing.assert_array_equal(numpy_helper.to_array(initializer),weights["actor."+initializer.name].numpy())
        subprocess.run([str(checker),str(target/"params/deploy.yaml"),str(validation/"synthetic_frames.txt"),
            str(target/"cpp_observations.txt"),str(policy),str(target/"cpp_actions.txt")],check=True)
        observations = np.loadtxt(target/"cpp_observations.txt",dtype=np.float32)
        np.testing.assert_array_equal(observations,np.loadtxt(preview/model/"cpp_observations.txt",dtype=np.float32))
        actor = actor_from_checkpoint(source,run,model,weights)
        with torch.inference_mode():
            expected = actor(torch.from_numpy(observations)).numpy()
        actual = np.loadtxt(target/"cpp_actions.txt",dtype=np.float32)
        assert actual.shape == (40,12) and np.isfinite(actual).all()
        np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-5)
        rows.append(dict(model=model,source_checkpoint=str(run/"model_4999.pt"),checkpoint_sha256=sha(run/"model_4999.pt"),
            policy_directory=str(target),onnx_sha256=sha(policy),input_dim=dim,output_dim=12,
            keyboard_lateral_speed=args.keyboard_lateral_speed,keyboard_yaw_speed=args.keyboard_yaw_speed,
            parameter_tensors_verified=len(graph.graph.initializer),cpp_history_frames_exact=40,
            max_action_error=float(np.abs(actual-expected).max())))
    assert all(sha(path) == value for path,value in protected.items())
    with (dest/"validation.csv").open("w",newline="") as handle:
        writer = csv.DictWriter(handle,list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (dest/"migration_manifest.json").write_text(json.dumps(dict(models=rows,protected_sha256=protected,
        controller_repository=str(repo),source_repository=str(source),
        controller_commit=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip(),
        note="Existing verified exports reused. CPU checks only; no DDS, simulator, training or checkpoint changes."),indent=2))
    print("MIGRATION_COMPLETE",dest)
    print(json.dumps(rows,indent=2))


if __name__ == "__main__":
    main()
