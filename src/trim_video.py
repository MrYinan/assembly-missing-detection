from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import cv2


def trim_video(input_path: Path, output_path: Path, start_sec: float, end_sec: float) -> None:
    if end_sec <= start_sec:
        raise ValueError("--end-sec must be greater than --start-sec")
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video metadata: {input_path}")

    start_frame = int(round(start_sec * fps))
    end_frame = int(round(end_sec * fps))
    if start_frame < 0 or end_frame > frame_count or start_frame >= end_frame:
        cap.release()
        duration = frame_count / fps
        raise ValueError(
            f"Trim range {start_sec:.2f}-{end_sec:.2f}s is outside video duration {duration:.2f}s"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot write video: {tmp_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    for _frame_idx in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1

    writer.release()
    cap.release()

    expected = end_frame - start_frame
    if written != expected:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Expected {expected} frames, wrote {written}")

    last_error: PermissionError | None = None
    for _attempt in range(8):
        try:
            shutil.copy2(tmp_path, output_path)
            try:
                tmp_path.unlink(missing_ok=True)
            except PermissionError:
                pass
            break
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.35)
    else:
        fallback_path = output_path.with_name(output_path.stem + "_new" + output_path.suffix)
        try:
            shutil.copy2(tmp_path, fallback_path)
        except PermissionError:
            fallback_path = tmp_path
        raise PermissionError(
            f"Cannot replace {output_path}. Close any video player or preview window using it. "
            f"The trimmed video is available at {fallback_path}."
        ) from last_error

    duration = written / fps
    print(
        f"Trimmed {input_path} [{start_sec:.2f}s, {end_sec:.2f}s) "
        f"-> {output_path} ({written} frames, {duration:.2f}s)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Trim an mp4 video by timestamp.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-sec", required=True, type=float)
    parser.add_argument("--end-sec", required=True, type=float)
    args = parser.parse_args()

    trim_video(args.input, args.output, args.start_sec, args.end_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
