#!/usr/bin/env python3
"""Plot Go2 posture metrics from lowstate CSV logs."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np


HIP_INDICES = np.array([0, 3, 6, 9])
THIGH_INDICES = np.array([1, 4, 7, 10])
CALF_INDICES = np.array([2, 5, 8, 11])

# SDK order: FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
#            RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
Q_REF_SDK = np.array(
    [
        -0.1,
        0.8,
        -1.5,
        0.1,
        0.8,
        -1.5,
        -0.1,
        1.0,
        -1.5,
        0.1,
        1.0,
        -1.5,
    ],
    dtype=float,
)

# FR and RR move outward in the negative hip direction; FL and RL in positive.
OUTWARD_SIGN = np.array([-1.0, 1.0, -1.0, 1.0])


def resolve_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(path) for path in glob.glob(pattern))
        if matches:
            paths.extend(matches)
            continue

        path = Path(pattern)
        if path.exists():
            paths.append(path)
        else:
            raise FileNotFoundError(f"No CSV files matched: {pattern}")

    unique_paths = sorted(dict.fromkeys(paths))
    if not unique_paths:
        raise FileNotFoundError("No CSV files were provided.")
    return unique_paths


def load_lowstate_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.dtype.names is None:
        raise ValueError(f"{path} is empty or has no CSV header.")

    required = ["time", *[f"q{i}" for i in range(12)]]
    missing = [name for name in required if name not in data.dtype.names]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    time = np.atleast_1d(data["time"]).astype(float)
    time = time - time[0]
    q = np.column_stack([np.atleast_1d(data[f"q{i}"]).astype(float) for i in range(12)])
    return time, q


def calculate_metrics(
    path: Path,
    q_ref: np.ndarray,
    start_time: float,
    end_time: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time, q = load_lowstate_csv(path)

    hip_delta = q[:, HIP_INDICES] - q_ref[HIP_INDICES]
    hip_outward = np.maximum(0.0, hip_delta * OUTWARD_SIGN)
    outward_metric = np.mean(hip_outward, axis=1)

    thigh_fold = np.maximum(0.0, q[:, THIGH_INDICES] - q_ref[THIGH_INDICES])
    calf_fold = np.maximum(0.0, q_ref[CALF_INDICES] - q[:, CALF_INDICES])
    folding_metric = (np.sum(thigh_fold, axis=1) + np.sum(calf_fold, axis=1)) / 8.0

    mask = time >= start_time
    if end_time is not None:
        mask &= time <= end_time
    if not np.any(mask):
        raise ValueError(f"{path} has no samples in the selected time window.")

    time = time[mask]
    time = time - time[0]
    return time, np.rad2deg(outward_metric[mask]), np.rad2deg(folding_metric[mask])


def align_runs(
    runs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(runs) == 1:
        time, outward, folding = runs[0]
        return time, outward[None, :], folding[None, :]

    start_time = max(run[0][0] for run in runs)
    end_time = min(run[0][-1] for run in runs)
    if end_time <= start_time:
        raise ValueError("Runs do not have an overlapping time range.")

    sample_count = min(len(run[0]) for run in runs)
    time = np.linspace(start_time, end_time, sample_count)
    outward = np.stack([np.interp(time, run[0], run[1]) for run in runs])
    folding = np.stack([np.interp(time, run[0], run[2]) for run in runs])
    return time - time[0], outward, folding


def parse_datasets(args: argparse.Namespace) -> list[tuple[str, list[Path]]]:
    datasets: list[tuple[str, list[Path]]] = []

    if args.baseline:
        datasets.append(("Baseline", resolve_paths(args.baseline)))
    if args.ours:
        datasets.append(("Ours", resolve_paths(args.ours)))
    if args.dataset:
        for item in args.dataset:
            label = item[0]
            patterns = item[1:]
            if not patterns:
                raise ValueError("--dataset requires a label and at least one CSV path or glob.")
            datasets.append((label, resolve_paths(patterns)))
    if args.csv:
        datasets.append(("Run", resolve_paths(args.csv)))

    if not datasets:
        raise ValueError("Provide --csv, --baseline/--ours, or one or more --dataset entries.")
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", help="CSV files or glob patterns for one unlabeled run set.")
    parser.add_argument("--baseline", nargs="+", help="Baseline CSV files or glob patterns.")
    parser.add_argument("--ours", nargs="+", help="Ours CSV files or glob patterns.")
    parser.add_argument(
        "--dataset",
        action="append",
        nargs="+",
        metavar=("LABEL", "CSV"),
        help="Custom dataset: --dataset LABEL file_or_glob [more_files_or_globs ...]",
    )
    parser.add_argument(
        "--reference",
        nargs=12,
        type=float,
        default=Q_REF_SDK,
        metavar="Q",
        help="12 reference joint angles in SDK order. Defaults to current Go2 deploy standing pose.",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="Ignore samples before this time in each CSV, in seconds. Use this to skip Passive/FixStand startup.",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="Ignore samples after this original CSV time, in seconds.",
    )
    parser.add_argument("--output", default="posture_metrics.png", help="Output image path.")
    parser.add_argument("--show", action="store_true", help="Show the figure after saving.")
    args = parser.parse_args()

    if args.start_time < 0.0:
        raise ValueError("--start-time must be non-negative.")
    if args.end_time is not None and args.end_time <= args.start_time:
        raise ValueError("--end-time must be greater than --start-time.")

    q_ref = np.asarray(args.reference, dtype=float)
    datasets = parse_datasets(args)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "Failed to import matplotlib. The current Python environment may have an incompatible "
            "numpy/matplotlib binary combination. Try running this script with your IsaacLab Python "
            "environment or reinstall matplotlib for the active numpy version."
        ) from exc

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for label, paths in datasets:
        runs = [calculate_metrics(path, q_ref, args.start_time, args.end_time) for path in paths]
        time, outward_runs, folding_runs = align_runs(runs)

        outward_mean = np.mean(outward_runs, axis=0)
        outward_std = np.std(outward_runs, axis=0)
        folding_mean = np.mean(folding_runs, axis=0)
        folding_std = np.std(folding_runs, axis=0)

        axes[0].plot(time, outward_mean, label=label)
        axes[0].fill_between(time, outward_mean - outward_std, outward_mean + outward_std, alpha=0.2)

        axes[1].plot(time, folding_mean, label=label)
        axes[1].fill_between(time, folding_mean - folding_std, folding_mean + folding_std, alpha=0.2)

    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Hip outward deviation (deg)")
    axes[0].set_title("(a) Hip outward deviation")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Leg folding deviation (deg)")
    axes[1].set_title("(b) Leg folding deviation")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    output = Path(args.output)
    if output.parent != Path("."):
        output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved {output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
