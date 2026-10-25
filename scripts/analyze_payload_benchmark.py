"""Recompute the 540 recorded Isaac Lab trials without changing source data."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path

from payload_metrics import read_csv, sha256, trial_metrics, write_csv, write_summaries, yaw_from_quaternion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.input_root.resolve(), args.output_dir.resolve()
    if output.exists():
        parser.error("Output exists; choose a new directory")
    protocol = json.loads((source / "protocol_v2.json").read_text())
    expected_models = ("Asym_PPO", "DreamWaQ_CENet", "PayloadEstimator")
    folders = sorted((source / "task1_v2").glob("*"))
    folders = [p for p in folders if p.is_dir()]
    if len(folders) != 18:
        raise ValueError(f"Expected 18 conditions, found {len(folders)}")
    output.mkdir(parents=True)
    protected = {str(source / "protocol_v2.json"): sha256(source / "protocol_v2.json")}
    episodes, validation, models, time_bins = [], [], {}, defaultdict(list)
    raw_rows, max_error = 0, 0.0
    with gzip.open(output / "source_trajectories.csv.gz", "wt", newline="", encoding="utf-8") as raw_handle:
        raw_writer = None
        for condition in folders:
            physical_reference = None
            for model in expected_models:
                folder = condition / model
                metadata = json.loads((folder / "manifest.json").read_text())
                if metadata["status"] != "complete" or not metadata["initial_pairing_verified"]:
                    raise ValueError(f"Incomplete/unpaired case: {folder}")
                if metadata["num_trials"] != 10 or metadata["duration"] != 12 or metadata["warmup"] != 2:
                    raise ValueError(f"Unexpected protocol: {folder}")
                checkpoint = metadata["checkpoint"]
                checkpoint_path = Path(checkpoint["checkpoint"])
                if sha256(checkpoint_path) != checkpoint["checkpoint_sha256"]:
                    raise ValueError(f"Checkpoint changed: {checkpoint_path}")
                models[model] = checkpoint
                protected[str(checkpoint_path)] = checkpoint["checkpoint_sha256"]
                for name in ("trajectory.csv", "episode_metrics.csv", "manifest.json", "initial_state.json", "config_overrides.json"):
                    path = folder / name
                    protected[str(path)] = sha256(path)
                initial = json.loads((folder / "initial_state.json").read_text())
                if physical_reference is None:
                    physical_reference = initial
                else:
                    for key in ("masses", "inertias", "coms", "materials", "joint_stiffness", "joint_damping", "joint_pos", "joint_vel", "command"):
                        if initial[key] != physical_reference[key]:
                            raise ValueError(f"Physical pairing mismatch: {folder}: {key}")
                    def differences(a, b):
                        if isinstance(a, list):
                            if len(a) != len(b):
                                raise ValueError("Paired shape mismatch")
                            return max((differences(x, y) for x, y in zip(a, b)), default=0)
                        return abs(float(a) - float(b))
                    for key in ("root_state", "full_policy_observation"):
                        if differences(initial[key], physical_reference[key]) > 1e-7:
                            raise ValueError(f"Initial pairing mismatch: {folder}: {key}")
                trials = defaultdict(list)
                for row in read_csv(folder / "trajectory.csv"):
                    if raw_writer is None:
                        raw_writer = csv.DictWriter(raw_handle, fieldnames=["condition"] + list(row))
                        raw_writer.writeheader()
                    raw_writer.writerow(dict(condition=condition.name, **row))
                    trials[int(row["trial"])].append(row)
                    raw_rows += 1
                    if row["model"] != model or float(row["true_mass_kg"]) != metadata["mass_kg"]:
                        raise ValueError(f"Raw identity mismatch: {folder}")
                    if abs(float(row["cmd_wz"]) - metadata["yaw_rate"]) > 1e-6:
                        raise ValueError(f"Command mismatch: {folder}")
                    for axis in ("vx", "vy", "wz"):
                        if abs(float(row[axis]) - float(row["cmd_" + axis]) - float(row[axis + "_error"])) > 1e-6:
                            raise ValueError(f"Stored error disagrees with state/command: {folder}")
                    if model == "PayloadEstimator":
                        time_bins[(metadata["terrain"], metadata["mass_kg"], metadata["yaw_rate"], int(float(row["time_before_s"]) * 10 + 1e-6))].append(
                            float(row["predicted_mass_kg"]))
                if set(trials) != set(range(10)):
                    raise ValueError(f"Trial IDs incomplete: {folder}")
                saved = {int(r["trial"]): r for r in read_csv(folder / "episode_metrics.csv")}
                for trial, rows in sorted(trials.items()):
                    if len(rows) != 600 or [int(r["step"]) for r in rows] != list(range(600)):
                        raise ValueError(f"Missing or duplicated samples: {folder}, trial {trial}")
                    if any(int(r["scored"]) != int(float(r["time_before_s"]) >= metadata["warmup"] - 1e-8) for r in rows):
                        raise ValueError(f"Scoring window mismatch: {folder}, trial {trial}")
                    yaw = yaw_from_quaternion(initial["root_state"][trial][3:7])
                    metrics = trial_metrics(rows, metadata["duration"], yaw, metadata["goal_x"])
                    check_fields = {k: k for k in ("vx_error_rmse", "vy_error_rmse", "linear_xy_vector_rmse", "wz_error_rmse", "roll_rms", "pitch_rms", "mass_mae_all_kg", "mass_mae_steady_kg", "mass_bias_steady_kg", "valid_runtime_s", "scored_runtime_s", "failed", "survived")}
                    check_fields["legacy_task_completed"] = "passed"
                    check_fields["legacy_task_incomplete"] = "incomplete"
                    delta = max((abs(float(metrics[k]) - float(saved[trial][v])) for k, v in check_fields.items() if metrics[k] != ""), default=0)
                    if delta > 1e-6:
                        raise ValueError(f"Metric mismatch {delta}: {folder}, trial {trial}")
                    max_error = max(max_error, delta)
                    episodes.append(dict(engine="IsaacLab_existing", condition=condition.name, model=model,
                                         terrain=metadata["terrain"], mass_kg=metadata["mass_kg"], cmd_wz=metadata["yaw_rate"],
                                         trial=trial, evaluation_seed=int(rows[0]["evaluation_seed"]), training_seed=int(rows[0]["training_seed"]), **metrics))
                    validation.append(dict(condition=condition.name, model=model, trial=trial, rows=len(rows), max_original_metric_difference=delta))
            print("Verified", condition.name, flush=True)
    if len(episodes) != 540 or raw_rows != 324000:
        raise ValueError(f"Unexpected counts: {len(episodes)}, {raw_rows}")
    if not all(sha256(path) == digest for path, digest in protected.items()):
        raise ValueError("Source changed during analysis")
    write_summaries(output, episodes)
    write_csv(output / "recalculation_checks.csv", validation)
    curves = [dict(terrain=k[0], mass_kg=k[1], cmd_wz=k[2], time_bin_start_s=k[3] / 10,
                   predicted_mass_kg_mean=statistics.mean(v), predicted_mass_kg_sd=statistics.stdev(v), samples=len(v))
              for k, v in sorted(time_bins.items())]
    write_csv(output / "mass_timecourse.csv", curves)
    (output / "manifest.json").write_text(json.dumps(dict(
        source_root=str(source), reused_rollouts=540, new_rollouts=0, raw_rows=raw_rows,
        protocol=protocol, checkpoints=models, protected_sha256=protected,
        max_original_metric_difference=max_error,
        units={"velocity": "m/s", "yaw_rate": "rad/s", "angle_rms": "rad", "angle_std_peak_mean": "degree", "mass": "kg"},
        limitations=["One training seed (42) per method; 10 paired initial states per condition.",
                     "SD describes trial or condition variability, not training-seed uncertainty.",
                     "Legacy completion uses maximum world-x progress >=3.2 m; interpret turning separately.",
                     "Planar reference integrates body velocity commands and initial yaw, ignoring slope pitch.",
                     "No observations or outcomes were discarded; failed/short trials must be interpreted with their runtime."]
    ), indent=2))
    print("ANALYSIS_COMPLETE", output, "trials", len(episodes), "rows", raw_rows, flush=True)


if __name__ == "__main__":
    main()
