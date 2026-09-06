#!/usr/bin/env python3

import argparse
import math
import os
import struct
import time
from pathlib import Path


WIDTH = 86
HEIGHT = 60
FRAME_VALUES = WIDTH * HEIGHT
FRAME_BYTES = FRAME_VALUES * 4


def clamp(value, low, high):
    return max(low, min(high, value))


def make_frame(frame_index, min_depth, max_depth):
    span = max_depth - min_depth
    values = []
    phase = frame_index * 0.17

    for y in range(HEIGHT):
        y_ratio = y / (HEIGHT - 1)
        for x in range(WIDTH):
            x_ratio = x / (WIDTH - 1)
            gradient = 0.75 * x_ratio + 0.25 * y_ratio
            wave = 0.08 * math.sin((x_ratio * 2.0 + phase) * math.pi)
            value = min_depth + span * clamp(gradient + wave, 0.0, 1.0)
            values.append(value)

    return values


def write_atomic(path, values):
    tmp_path = path.with_name(path.name + ".tmp")
    payload = struct.pack(f"<{len(values)}f", *values)
    if len(payload) != FRAME_BYTES:
        raise RuntimeError(f"depth payload has {len(payload)} bytes, expected {FRAME_BYTES}")

    with tmp_path.open("wb") as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser(description="Write fake PIE 60x86 float32 depth frames atomically.")
    parser.add_argument("--path", type=Path, default=Path("/tmp/pie_depth_60x86_float32.bin"))
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--min-depth", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=1.0)
    parser.add_argument("--frames", type=int, default=0, help="0 means run forever.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--print-every", type=int, default=20)
    args = parser.parse_args()

    if args.hz <= 0.0:
        raise ValueError("--hz must be positive")
    if args.min_depth >= args.max_depth:
        raise ValueError("--min-depth must be smaller than --max-depth")

    args.path.parent.mkdir(parents=True, exist_ok=True)
    frame_index = 0
    period = 1.0 / args.hz
    next_time = time.monotonic()
    max_frames = 1 if args.once else args.frames

    print(f"writing {WIDTH}x{HEIGHT} float32 depth to {args.path}")
    print(f"expected file size: {FRAME_BYTES} bytes")

    while max_frames == 0 or frame_index < max_frames:
        values = make_frame(frame_index, args.min_depth, args.max_depth)
        write_atomic(args.path, values)

        if args.print_every > 0 and frame_index % args.print_every == 0:
            print(
                "frame={:06d} bytes={} min={:.3f} max={:.3f} samples=[{:.3f}, {:.3f}, {:.3f}]".format(
                    frame_index,
                    args.path.stat().st_size,
                    min(values),
                    max(values),
                    values[0],
                    values[100],
                    values[-1],
                ),
                flush=True,
            )

        frame_index += 1
        if args.once:
            break

        next_time += period
        sleep_time = next_time - time.monotonic()
        if sleep_time > 0.0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
