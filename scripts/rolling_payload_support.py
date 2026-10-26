"""Physics and statistics for an independent rolling-payload evaluation."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from payload_metrics import trial_metrics, write_csv

MODELS = ("Asym_PPO", "DreamWaQ_CENet", "PayloadEstimator")
METRICS = ("roll_rms", "flatness_penalty", "linear_xy_vector_rmse")
JOINTS = [f"{leg}_{kind}_joint" for kind in ("hip", "thigh", "calf")
          for leg in ("FL", "FR", "RL", "RR")]
WIDTHS = (3, 3, 3, 12, 12, 12)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str)+"\n")


def build_scene(original, directory, mass):
    """Write evaluation-only copies; infer sphere inertia from its geom mass.

    All robot masses, explicit base inertia and collision settings are preserved.
    Only sphere mass, nonpenetrating default sphere height and asset/include paths
    change. The original files are read only; no runtime mass edits are used.
    """
    import mujoco
    original = Path(original).resolve()
    directory = Path(directory)
    directory.mkdir()
    scene = ET.parse(original)
    include = scene.getroot().find("include")
    robot_path = original.parent/include.get("file")
    robot = ET.parse(robot_path)
    compiler = robot.getroot().find("compiler")
    compiler.set("meshdir", str((robot_path.parent/compiler.get("meshdir", ".")).resolve()))
    base = robot.find(".//body[@name='base_link']")
    ball = robot.find(".//body[@name='payload_ball']")
    assert base is not None and ball is not None and ball.find("freejoint") is not None
    geom = ball.find("geom[@name='ball_geom']")
    assert geom.get("type") == "sphere" and ball.find("inertial") is None
    radius = float(geom.get("size"))
    geom.set("mass", str(float(mass)))
    xyz = [float(v) for v in base.get("pos").split()]
    xyz[2] += .057+radius+.002
    ball.set("pos", " ".join(map(str, xyz)))
    robot.write(directory/"robot.xml", encoding="unicode")
    include.set("file", "robot.xml")
    scene.write(directory/"scene.xml", encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(directory/"scene.xml"))
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload_ball")
    np.testing.assert_allclose(model.body_mass[base_id], 6.921, rtol=0, atol=1e-12)
    np.testing.assert_allclose(model.body_mass[ball_id], mass, rtol=0, atol=1e-12)
    np.testing.assert_allclose(model.body_inertia[ball_id], .4*mass*radius**2, rtol=0, atol=1e-12)
    assert model.jnt_type[model.body_jntadr[ball_id]] == mujoco.mjtJoint.mjJNT_FREE
    assert model.body_parentid[ball_id] == 0 and model.nu == 12
    info = dict(source_scene=str(original), source_robot=str(robot_path), sphere_mass_kg=mass,
        radius_m=radius, sphere_inertia=model.body_inertia[ball_id].tolist(),
        base_mass_kg=float(model.body_mass[base_id]), total_mass_kg=float(model.body_mass.sum()),
        base_inertia=model.body_inertia[base_id].tolist(), base_com=model.body_ipos[base_id].tolist(),
        physics_dt=float(model.opt.timestep), nq=model.nq, nv=model.nv, nu=model.nu,
        body_masses=model.body_mass.tolist(), body_inertias=model.body_inertia.tolist(),
        geom_friction=model.geom_friction.tolist(), free_ball=True,
        overrides="Sphere geom mass, nonpenetrating default height, absolute mesh paths and local include path only")
    write_json(directory/"physics.json", info)
    return model, info


def initial_samples(seed, count):
    rows = []
    for trial in range(count):
        trial_seed = int(np.random.SeedSequence([seed, trial]).generate_state(1)[0])
        rng = np.random.default_rng(trial_seed)
        rows.append(dict(trial=trial, evaluation_seed=trial_seed,
            xy=rng.uniform([-.05,-.08],[.05,.08]).tolist(),
            rpy=rng.uniform([-.02,-.02,-.1],[.02,.02,.1]).tolist(),
            joint_noise=rng.uniform(-.02,.02,12).tolist(),
            joint_velocity=rng.uniform(-.02,.02,12).tolist()))
    return rows


def observation(frame, history, length):
    """Current frame followed by histories in term order, oldest to newest."""
    if not length:
        return frame.copy(), []
    history = history[1:]+[frame.copy()] if history else [frame.copy() for _ in range(length)]
    frames = np.stack(history)
    offset, terms = 0, []
    for width in WIDTHS:
        terms.append(frames[:,offset:offset+width].reshape(-1))
        offset += width
    return np.concatenate([frame]+terms), history


def indices(model):
    import mujoco
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload_ball")
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINTS]
    assert min(joints) >= 0
    actuator = np.array([np.flatnonzero(model.actuator_trnid[:,0] == j).item() for j in joints])
    ball_joint = model.body_jntadr[ball]
    return dict(base=base, ball=ball, qidx=model.jnt_qposadr[joints], vidx=model.jnt_dofadr[joints],
        actuator=actuator, ball_qidx=int(model.jnt_qposadr[ball_joint]),
        ball_vidx=int(model.jnt_dofadr[ball_joint]),
        ball_geom=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"ball_geom"))


def ball_contacts(model, data, idx):
    import mujoco
    robot_contacts, floor_contacts, minimum_gap, normal_force = 0, 0, math.inf, 0.
    for contact_index, contact in enumerate(data.contact):
        pair = (int(contact.geom1),int(contact.geom2))
        if idx["ball_geom"] not in pair:
            continue
        other = pair[1] if pair[0] == idx["ball_geom"] else pair[0]
        body = int(model.geom_bodyid[other])
        minimum_gap = min(minimum_gap,float(contact.dist))
        if body == 0:
            floor_contacts += 1
        elif model.body_rootid[body] == idx["base"]:
            robot_contacts += 1
            force = np.zeros(6)
            mujoco.mj_contactForce(model,data,contact_index,force)
            normal_force += max(0.,float(force[0]))
    return dict(ball_robot_contacts=robot_contacts,ball_floor_contacts=floor_contacts,
                ball_min_contact_gap_m=minimum_gap if math.isfinite(minimum_gap) else "",
                ball_robot_normal_force_sum_n=normal_force)


def state_values(model, data, idx):
    rotation = data.xmat[idx["base"]].reshape(3,3)
    body_velocity = rotation.T@data.qvel[:3]
    body_omega = data.sensor("imu_gyro").data.copy()
    gravity = rotation.T@np.array([0.,0.,-1.])
    relative = rotation.T@(data.xpos[idx["ball"]]-data.xpos[idx["base"]])
    ball_velocity = data.qvel[idx["ball_vidx"]:idx["ball_vidx"]+3]
    relative_velocity = rotation.T@(ball_velocity-data.qvel[:3])-np.cross(body_omega,relative)
    w,x,y,z = data.qpos[3:7]
    roll = math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
    pitch = math.asin(float(np.clip(2*(w*y-z*x),-1,1)))
    row = dict(vx=float(body_velocity[0]),vy=float(body_velocity[1]),wz=float(body_omega[2]),
        roll=roll,pitch=pitch,qw=float(w),qx=float(x),qy=float(y),qz=float(z),
        gx=float(gravity[0]),gy=float(gravity[1]),gz=float(gravity[2]),
        base_x=float(data.qpos[0]),base_y=float(data.qpos[1]),height=float(data.qpos[2]),
        ball_rel_x=float(relative[0]),ball_rel_y=float(relative[1]),ball_rel_z=float(relative[2]),
        ball_rel_vx=float(relative_velocity[0]),ball_rel_vy=float(relative_velocity[1]),ball_rel_vz=float(relative_velocity[2]),
        ball_x=float(data.xpos[idx["ball"],0]),ball_y=float(data.xpos[idx["ball"],1]),ball_z=float(data.xpos[idx["ball"],2]),
        ball_wx=float(data.qvel[idx["ball_vidx"]+3]),ball_wy=float(data.qvel[idx["ball_vidx"]+4]),
        ball_wz=float(data.qvel[idx["ball_vidx"]+5]),tilt_cosine=float(rotation[2,2]),
        actual_base_mass_kg=float(model.body_mass[idx["base"]]),true_mass_kg=float(model.body_mass[idx["ball"]]),
        **ball_contacts(model,data,idx))
    assert abs(np.linalg.norm(gravity)-1.) < 1e-8
    return row


class CargoMonitor:
    """Loss: ball touches floor, or fully outside tray footprint for >=0.1 s."""
    def __init__(self, dt=.02):
        self.dt, self.outside_steps, self.first_loss_s = dt, 0, None

    def update(self, values, time_s):
        outside = abs(values["ball_rel_x"]) > .20 or abs(values["ball_rel_y"]) > .15
        self.outside_steps = self.outside_steps+1 if outside else 0
        if self.first_loss_s is None and (values["ball_floor_contacts"] or self.outside_steps*self.dt >= .1-1e-9):
            self.first_loss_s = time_s
        return int(self.first_loss_s is not None)


def metrics(rows, duration=12., warmup=2.):
    """Retain all trials. Main scores stop at cargo loss; full-window scores are auxiliary."""
    selected = [r for r in rows if float(r["time_before_s"]) >= warmup-1e-9]
    carried = [r for r in selected if not int(r["cargo_lost"])]
    invalid = sum(not all(math.isfinite(float(r[k])) for k in
                  ("roll","vx","vy","gx","gy","gz")) for r in rows)
    norm_error = max((abs(math.sqrt(sum(float(r[k])**2 for k in ("gx","gy","gz")))-1.) for r in rows),default=0.)
    invalid += int(norm_error > 1e-8)
    result = dict(samples=len(rows),scored_samples=len(carried),full_window_samples=len(selected),
        observed_s=float(rows[-1]["time_after_s"]) if rows else 0.,invalid_samples=invalid,
        max_gravity_norm_error=norm_error,failed=int(any(int(r["terminated"]) for r in rows)),
        cargo_lost=int(any(int(r["cargo_lost"]) for r in rows)),
        first_loss_s=next((float(r["time_after_s"]) for r in rows if int(r["cargo_lost"])),""))
    result["robot_completed"] = int(bool(rows) and not result["failed"] and result["observed_s"] >= duration-1e-6)
    result["transport_completed"] = int(result["robot_completed"] and not result["cargo_lost"] and not invalid)
    result["censored"] = int(not result["transport_completed"])
    for prefix, subset in (("",carried),("full_window_",selected)):
        values = dict.fromkeys(METRICS, "")
        if subset and not invalid:
            # Reuse existing roll/velocity definitions; only flatness is new here.
            existing = trial_metrics(subset,duration)
            values = dict(roll_rms=existing["roll_rms"],linear_xy_vector_rmse=existing["linear_xy_vector_rmse"],
                          flatness_penalty=statistics.mean(float(r["gx"])**2+float(r["gy"])**2 for r in subset))
        result.update({prefix+k:v for k,v in values.items()})
    speeds = [float(r["vx"]) for r in carried]
    longest = consecutive = 0
    for v in speeds:
        consecutive = consecutive+1 if v < .1 else 0
        longest = max(longest,consecutive)
    result.update(mean_vx=statistics.mean(speeds) if speeds else "",stall_flag=int(bool(speeds) and
        (statistics.mean(speeds)<.1 or longest*.02>=2.)),
        final_x_m=float(rows[-1]["x_progress"]) if rows else "",
        max_ball_planar_offset_m=max((math.hypot(float(r["ball_rel_x"]),float(r["ball_rel_y"])) for r in rows),default=0.),
        ball_relative_path_m=sum(math.sqrt(sum((float(b[k])-float(a[k]))**2 for k in
            ("ball_rel_x","ball_rel_y","ball_rel_z"))) for a,b in zip(rows,rows[1:])),
        mass_mae_steady_kg=statistics.mean(abs(float(r["predicted_mass_kg"])-float(r["true_mass_kg"])) for r in carried)
            if carried and carried[0].get("predicted_mass_kg","") != "" else "")
    return result


def aggregate(episodes):
    result = dict(trials=len(episodes),robot_completed=sum(int(r["robot_completed"]) for r in episodes),
        transport_completed=sum(int(r["transport_completed"]) for r in episodes),
        cargo_losses=sum(int(r["cargo_lost"]) for r in episodes),robot_failures=sum(int(r["failed"]) for r in episodes),
        stalls=sum(int(r["stall_flag"]) for r in episodes),invalid_trials=sum(int(r["invalid_samples"])>0 for r in episodes),
        scored_samples_min=min(int(r["scored_samples"]) for r in episodes),
        scored_samples_max=max(int(r["scored_samples"]) for r in episodes))
    for name in (*METRICS,"mean_vx","max_ball_planar_offset_m","ball_relative_path_m","mass_mae_steady_kg"):
        values = [float(r[name]) for r in episodes if r[name] != "" and math.isfinite(float(r[name]))]
        result.update({name+"_mean":statistics.mean(values) if values else "",
                       name+"_sd":statistics.stdev(values) if len(values)>1 else "",name+"_n":len(values)})
    return result
