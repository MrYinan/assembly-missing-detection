# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def open_video(path: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open video: {path}')
    return cap


def video_info(path: str | Path) -> Dict[str, float]:
    cap = open_video(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        'fps': fps,
        'frames': frames,
        'width': w,
        'height': h,
        'duration_sec': frames / fps if fps else 0.0,
    }


def iter_video_samples(video_path: str | Path, start_sec: float, end_sec: float | None, sample_fps: float):
    cap = open_video(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_f = max(0, int(round(start_sec * fps)))
    end_f = total if end_sec is None or end_sec < 0 else min(total, int(round(end_sec * fps)))
    step = max(1, int(round(fps / max(sample_fps, 1e-6))))
    for frame_idx in range(start_f, end_f, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        yield frame_idx, frame_idx / fps, frame
    cap.release()


def draw_title(img: np.ndarray, text: str, org=(18, 36), scale=0.82) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (35, 35, 35), 1, cv2.LINE_AA)


def side_by_side(left: np.ndarray, right: np.ndarray, left_title='Original', right_title='Deep PatchCore') -> np.ndarray:
    h, w = left.shape[:2]
    right = cv2.resize(right, (w, h), interpolation=cv2.INTER_AREA)
    l = left.copy(); r = right.copy()
    draw_title(l, left_title)
    draw_title(r, right_title)
    return np.concatenate([l, r], axis=1)


def group_alarm_segments(times_or_rows, flags=None, min_samples: int = 1, merge_gap_sec: float = 2.0) -> List[Dict]:
    """Group NG samples into continuous alarm segments.

    Supports both forms:
    - group_alarm_segments(frame_rows, merge_gap_sec=..., min_samples=...)
    - group_alarm_segments(times, flags, merge_gap_sec=..., min_samples=...)
    """
    if flags is None:
        rows = sorted([r for r in times_or_rows if str(r.get('frame_label', 'OK')).upper() == 'NG'], key=lambda r: float(r['time_sec']))
        ng_rows = rows
    else:
        ng_rows = [{'time_sec': float(t)} for t, f in zip(times_or_rows, flags) if bool(f)]
        ng_rows = sorted(ng_rows, key=lambda r: float(r['time_sec']))
    if not ng_rows:
        return []
    segs = []
    cur = [ng_rows[0]]
    for r in ng_rows[1:]:
        if float(r['time_sec']) - float(cur[-1]['time_sec']) <= merge_gap_sec:
            cur.append(r)
        else:
            if len(cur) >= min_samples:
                segs.append(_segment(cur))
            cur = [r]
    if len(cur) >= min_samples:
        segs.append(_segment(cur))
    return segs


def _segment(rows: List[Dict]) -> Dict:
    ts = [float(r['time_sec']) for r in rows]
    return {
        'start_sec': float(min(ts)),
        'end_sec': float(max(ts)),
        'duration_sec': float(max(ts) - min(ts)),
        'sample_count': int(len(rows)),
        'max_product_score': float(max(float(r.get('max_product_score', 0.0)) for r in rows)),
        'ng_sample_count': int(sum(int(r.get('ng_product_count', 0)) for r in rows)),
    }


def bgr_to_rgb(crop: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
