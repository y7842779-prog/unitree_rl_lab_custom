#!/usr/bin/env python3

"""Bridge unitree_mujoco's live Parkour depth file into PIE's depth protocol.

This is a stage-2 validation bridge: it proves the C++ PIE runtime can consume
live MuJoCo depth updates. It does not replace the final PIE-aligned renderer,
because unitree_mujoco's current Parkour depth output uses a different camera
model and image shape.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np


SOURCE_HEIGHT = 87
SOURCE_WIDTH = 58
PIE_HEIGHT = 60
PIE_WIDTH = 86
SOURCE_BYTES = SOURCE_HEIGHT * SOURCE_WIDTH * np.dtype(np.float32).itemsize
PIE_BYTES = PIE_HEIGHT * PIE_WIDTH * np.dtype(np.float32).itemsize


def read_source_frame(path: Path) -> np.ndarray | None:
    try:
        if path.stat().st_size != SOURCE_BYTES:
            return None
        frame = np.fromfile(path, dtype=np.float32)
    except FileNotFoundError:
        return None

    if frame.size != SOURCE_HEIGHT * SOURCE_WIDTH:
        return None
    if not np.all(np.isfinite(frame)):
        return None
    return frame.reshape(SOURCE_HEIGHT, SOURCE_WIDTH)


def resize_bilinear(image: np.ndarray, out_height: int, out_width: int) -> np.ndarray:
    in_height, in_width = image.shape
    if in_height == out_height and in_width == out_width:
        return image.astype(np.float32, copy=True)

    y = np.linspace(0.0, in_height - 1, out_height, dtype=np.float32)
    x = np.linspace(0.0, in_width - 1, out_width, dtype=np.float32)
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, in_height - 1)
    x1 = np.clip(x0 + 1, 0, in_width - 1)
    wy = (y - y0).reshape(-1, 1)
    wx = (x - x0).reshape(1, -1)

    top = image[y0[:, None], x0[None, :]] * (1.0 - wx) + image[y0[:, None], x1[None, :]] * wx
    bottom = image[y1[:, None], x0[None, :]] * (1.0 - wx) + image[y1[:, None], x1[None, :]] * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32)


def convert_to_pie_depth(
    source_norm: np.ndarray,
    *,
    source_max_depth: float,
    pie_min_depth: float,
    pie_max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    source_m = np.clip(source_norm + 0.5, 0.0, 1.0) * source_max_depth
    resized_m = resize_bilinear(source_m, PIE_HEIGHT, PIE_WIDTH)
    pie_norm = np.clip(resized_m, pie_min_depth, pie_max_depth) / pie_max_depth
    return resized_m.astype(np.float32), np.ascontiguousarray(pie_norm, dtype=np.float32)


def write_atomic(path: Path, frame: np.ndarray) -> None:
    payload = np.ascontiguousarray(frame, dtype=np.float32).tobytes(order="C")
    if len(payload) != PIE_BYTES:
        raise RuntimeError(f"PIE depth payload has {len(payload)} bytes, expected {PIE_BYTES}")

    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp_path, path)


def make_self_test_source(frame_index: int) -> np.ndarray:
    y = np.linspace(0.0, 1.0, SOURCE_HEIGHT, dtype=np.float32).reshape(-1, 1)
    x = np.linspace(0.0, 1.0, SOURCE_WIDTH, dtype=np.float32).reshape(1, -1)
    wave = 0.1 * np.sin((x * 2.0 + frame_index * 0.15) * math.pi).astype(np.float32)
    meters = np.clip(0.35 + 1.2 * (0.7 * x + 0.3 * y) + wave, 0.0, 2.0)
    return meters / 2.0 - 0.5


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert live unitree_mujoco depth into PIE 60x86 depth.")
    parser.add_argument("--source-path", type=Path, default=Path("/tmp/parkour_depth_image_float32.bin"))
    parser.add_argument("--target-path", type=Path, default=Path("/tmp/pie_depth_60x86_float32.bin"))
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--source-max-depth", type=float, default=2.0)
    parser.add_argument("--pie-min-depth", type=float, default=0.05)
    parser.add_argument("--pie-max-depth", type=float, default=3.0)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--frames", type=int, default=0, help="0 means run forever.")
    parser.add_argument("--self-test", action="store_true", help="Generate synthetic source depth instead of reading a file.")
    args = parser.parse_args()

    if args.hz <= 0.0:
        raise ValueError("--hz must be positive")
    if args.source_max_depth <= 0.0 or args.pie_max_depth <= 0.0:
        raise ValueError("depth ranges must be positive")
    if args.pie_min_depth >= args.pie_max_depth:
        raise ValueError("--pie-min-depth must be smaller than --pie-max-depth")

    args.target_path.parent.mkdir(parents=True, exist_ok=True)
    period = 1.0 / args.hz
    next_time = time.monotonic()
    frame_index = 0

    print(f"source: {args.source_path} [{SOURCE_HEIGHT}x{SOURCE_WIDTH}, {SOURCE_BYTES} bytes]")
    print(f"target: {args.target_path} [{PIE_HEIGHT}x{PIE_WIDTH}, {PIE_BYTES} bytes]")

    while args.frames == 0 or frame_index < args.frames:
        source_norm = make_self_test_source(frame_index) if args.self_test else read_source_frame(args.source_path)
        if source_norm is not None:
            depth_m, pie_depth = convert_to_pie_depth(
                source_norm,
                source_max_depth=args.source_max_depth,
                pie_min_depth=args.pie_min_depth,
                pie_max_depth=args.pie_max_depth,
            )
            write_atomic(args.target_path, pie_depth)

            if args.print_every > 0 and frame_index % args.print_every == 0:
                flat = pie_depth.reshape(-1)
                print(
                    "frame={:06d} source_norm=[{:.3f}, {:.3f}] depth_m=[{:.3f}, {:.3f}] "
                    "pie=[{:.3f}, {:.3f}] bytes={} samples=[{:.3f}, {:.3f}, {:.3f}]".format(
                        frame_index,
                        float(source_norm.min()),
                        float(source_norm.max()),
                        float(depth_m.min()),
                        float(depth_m.max()),
                        float(pie_depth.min()),
                        float(pie_depth.max()),
                        args.target_path.stat().st_size,
                        float(flat[0]),
                        float(flat[100]),
                        float(flat[-1]),
                    ),
                    flush=True,
                )
        elif args.print_every > 0 and frame_index % args.print_every == 0:
            print(f"frame={frame_index:06d} waiting for complete source frame", flush=True)

        frame_index += 1
        next_time += period
        sleep_time = next_time - time.monotonic()
        if sleep_time > 0.0:
            time.sleep(sleep_time)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
