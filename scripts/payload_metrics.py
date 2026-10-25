"""Shared trial metrics for recorded Isaac Lab and isolated MuJoCo evaluations."""

from __future__ import annotations

import csv
import hashlib
import math
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = (
    "vx_error_rmse", "vy_error_rmse", "linear_xy_vector_rmse", "wz_error_rmse",
    "roll_rms", "pitch_rms", "roll_std_deg", "pitch_std_deg",
    "roll_mean_deg", "pitch_mean_deg", "roll_peak_abs_deg", "pitch_peak_abs_deg",
    "mean_vx", "mean_vy", "mean_wz", "mass_mae_all_kg", "mass_mae_steady_kg",
    "mass_rmse_steady_kg", "mass_bias_steady_kg", "valid_runtime_s", "scored_runtime_s",
    "final_x_m", "final_y_m", "max_x_m", "path_xy_m",
    "reference_position_rmse_m", "reference_endpoint_error_m",
)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def yaw_from_quaternion(quaternion):
    w, x, y, z = quaternion
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def trial_metrics(rows, duration, initial_yaw=0.0, legacy_goal_x=None):
    """Score samples marked `scored`; preserve failures and observed durations."""
    if not rows:
        raise ValueError("Empty trial")
    scored = [row for row in rows if int(row["scored"])]

    def values(key, selection=scored):
        result = [float(row[key]) for row in selection]
        if not all(math.isfinite(value) for value in result):
            raise ValueError(f"Nonfinite {key}")
        return result

    errors = {axis: [float(r[axis]) - float(r["cmd_" + axis]) for r in scored]
              for axis in ("vx", "vy", "wz")}
    result = {axis + "_error_rmse": math.sqrt(statistics.mean(v * v for v in errors[axis])) if errors[axis] else ""
              for axis in errors}
    result["linear_xy_vector_rmse"] = math.sqrt(statistics.mean(
        x * x + y * y for x, y in zip(errors["vx"], errors["vy"]))) if scored else ""
    for angle in ("roll", "pitch"):
        data = values(angle)
        result[angle + "_rms"] = math.sqrt(statistics.mean(v * v for v in data)) if data else ""
        result[angle + "_std_deg"] = math.degrees(statistics.pstdev(data)) if data else ""
        result[angle + "_mean_deg"] = math.degrees(statistics.mean(data)) if data else ""
        result[angle + "_peak_abs_deg"] = math.degrees(max(map(abs, data))) if data else ""
    for axis in ("vx", "vy", "wz"):
        result["mean_" + axis] = statistics.mean(values(axis)) if scored else ""
    for key in ("mass_mae_all_kg", "mass_mae_steady_kg", "mass_rmse_steady_kg", "mass_bias_steady_kg"):
        result[key] = ""
    predictions = [r.get("predicted_mass_kg", "") for r in rows]
    if any(v not in ("", "nan", "None") for v in predictions):
        all_errors = [float(r["predicted_mass_kg"]) - float(r["true_mass_kg"]) for r in rows]
        mass_errors = [float(r["predicted_mass_kg"]) - float(r["true_mass_kg"]) for r in scored]
        if not all(math.isfinite(v) for v in all_errors):
            raise ValueError("Nonfinite mass prediction")
        result.update(mass_mae_all_kg=statistics.mean(map(abs, all_errors)),
                      mass_mae_steady_kg=statistics.mean(map(abs, mass_errors)) if mass_errors else "",
                      mass_rmse_steady_kg=math.sqrt(statistics.mean(v * v for v in mass_errors)) if mass_errors else "",
                      mass_bias_steady_kg=statistics.mean(mass_errors) if mass_errors else "")
    failed = any(int(r["terminated"]) for r in rows)
    observed = float(rows[-1]["time_after_s"])
    survived = not failed and observed >= duration - 1e-6
    result.update(scored_samples=len(scored), failed=int(failed), survived=int(survived),
                  timeout=int(any(int(r["timeout"]) for r in rows)), valid_runtime_s=observed,
                  scored_runtime_s=sum(float(r["time_after_s"]) - float(r["time_before_s"]) for r in scored),
                  final_x_m=float(rows[-1]["x_progress"]), final_y_m=float(rows[-1]["y_progress"]),
                  max_x_m=max(values("x_progress", rows)), legacy_task_completed="", legacy_task_incomplete="")
    if legacy_goal_x is not None:
        complete = survived and result["max_x_m"] >= legacy_goal_x
        result.update(legacy_task_completed=int(complete), legacy_task_incomplete=int(survived and not complete))
    reference_x = reference_y = previous_x = previous_y = path = 0.0
    heading = initial_yaw
    reference_errors = []
    for row in rows:
        dt = float(row["time_after_s"]) - float(row["time_before_s"])
        if dt <= 0:
            raise ValueError("Nonpositive sample duration")
        vx, vy, wz = (float(row["cmd_" + axis]) for axis in ("vx", "vy", "wz"))
        if abs(wz) < 1e-12:
            dx, dy = vx * dt, vy * dt
        else:
            s, c = math.sin(wz * dt), math.cos(wz * dt)
            dx, dy = (vx * s + vy * (c - 1)) / wz, (vx * (1 - c) + vy * s) / wz
        reference_x += math.cos(heading) * dx - math.sin(heading) * dy
        reference_y += math.sin(heading) * dx + math.cos(heading) * dy
        heading += wz * dt
        x, y = float(row["x_progress"]), float(row["y_progress"])
        path += math.hypot(x - previous_x, y - previous_y)
        previous_x, previous_y = x, y
        if int(row["scored"]):
            reference_errors.append((x - reference_x) ** 2 + (y - reference_y) ** 2)
    result.update(path_xy_m=path, reference_position_rmse_m=math.sqrt(statistics.mean(reference_errors)) if reference_errors else "",
                  reference_endpoint_error_m=math.hypot(previous_x - reference_x, previous_y - reference_y))
    return result


def summarize(episodes, group_keys):
    groups = defaultdict(list)
    for row in episodes:
        groups[tuple(row[k] for k in group_keys)].append(row)
    results = []
    for key, rows in sorted(groups.items()):
        result = dict(zip(group_keys, key))
        result["trials"] = len(rows)
        for field, label in (("survived", "survival"), ("failed", "failure"),
                             ("legacy_task_completed", "legacy_task_completion"),
                             ("legacy_task_incomplete", "legacy_task_incomplete")):
            data = [float(r[field]) for r in rows if r.get(field, "") != ""]
            result[label + "_count"] = int(sum(data)) if data else ""
            result[label + "_rate"] = statistics.mean(data) if data else ""
        for metric in METRICS:
            data = [float(r[metric]) for r in rows if r.get(metric, "") != ""]
            result[metric + "_mean"] = statistics.mean(data) if data else ""
            result[metric + "_sd"] = statistics.stdev(data) if len(data) > 1 else (0.0 if data else "")
            result[metric + "_n"] = len(data)
        results.append(result)
    return results


def write_summaries(directory, episodes):
    directory = Path(directory)
    write_csv(directory / "episode_metrics.csv", episodes)
    for filename, keys in (("overall_summary.csv", ("model",)),
                           ("terrain_summary.csv", ("terrain", "model")),
                           ("load_summary.csv", ("mass_kg", "model")),
                           ("condition_summary.csv", ("terrain", "mass_kg", "cmd_wz", "model"))):
        write_csv(directory / filename, summarize(episodes, keys))
