# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from .detectors import make_detector, load_detector
from .utils import ensure_dir, load_json, save_json, iter_video_samples, side_by_side, group_alarm_segments, video_info


def _is_time_in_segments(t: float, segments: List[Dict]) -> bool:
    return any(float(seg["start_sec"]) <= float(t) <= float(seg["end_sec"]) for seg in segments)


def roi_to_xywh(roi):
    """Convert ROI formats to x, y, w, h.

    Video B returns rectangle ROI as [x, y, w, h].
    Video A returns circular ROI as [cx, cy, r].
    The pipeline CSV/render code always uses x/y/w/h, so circle ROIs are
    converted to their enclosing square.
    """
    if roi is None:
        return 0, 0, 0, 0
    if isinstance(roi, dict):
        if all(k in roi for k in ("x", "y", "w", "h")):
            return int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])
        if all(k in roi for k in ("cx", "cy", "r")):
            cx, cy, r = float(roi["cx"]), float(roi["cy"]), float(roi["r"])
            return int(cx - r), int(cy - r), int(2 * r), int(2 * r)
    try:
        vals = list(roi)
    except TypeError:
        return 0, 0, 0, 0
    if len(vals) == 4:
        x, y, w, h = vals
        return int(x), int(y), int(w), int(h)
    if len(vals) == 3:
        cx, cy, r = vals
        return int(float(cx) - float(r)), int(float(cy) - float(r)), int(2 * float(r)), int(2 * float(r))
    if len(vals) > 4:
        x, y, w, h = vals[:4]
        return int(x), int(y), int(w), int(h)
    return 0, 0, 0, 0


def _accepted_segments_from_frame_df(frame_df: pd.DataFrame, *, min_samples: int, merge_gap_sec: float,
                                     strong_score_threshold: float | None = None) -> List[Dict]:
    """Build final alarm segments from raw frame predictions.

    The detector first marks candidate NG frames by comparing product scores with the normal memory-bank threshold.
    Industrial video often contains one-frame reflections or short transition noise.  This function performs a
    non-cheating, time-agnostic post-processing step:
      1. merge temporally adjacent candidate NG frames;
      2. discard segments with too few samples;
      3. optionally keep only segments whose maximum anomaly score is strong enough.

    No manual defect timestamp is used here.  Manual labels are evaluated only later in evaluate_keyframes.py.
    """
    if frame_df.empty:
        return []
    if "raw_frame_label" not in frame_df.columns:
        # This can happen when the test range contains no sampled frame.
        # Return no alarms instead of crashing, so the caller can inspect
        # sampled_frames/test_start/test_end in the summary.
        return []
    raw_flags = (frame_df["raw_frame_label"] == "NG").tolist()
    raw_segments = group_alarm_segments(frame_df["time_sec"].tolist(), raw_flags,
                                        min_samples=min_samples, merge_gap_sec=merge_gap_sec)
    accepted: List[Dict] = []
    for seg in raw_segments:
        m = (frame_df["time_sec"] >= seg["start_sec"]) & (frame_df["time_sec"] <= seg["end_sec"])
        seg = dict(seg)
        seg["max_product_score"] = float(frame_df.loc[m, "max_product_score"].max()) if m.any() else 0.0
        seg["raw_ng_sample_count"] = int((frame_df.loc[m, "raw_frame_label"] == "NG").sum()) if m.any() else 0
        if strong_score_threshold is not None and seg["max_product_score"] < float(strong_score_threshold):
            seg["accepted"] = False
            seg["reject_reason"] = "weak_segment_max_score"
            continue
        seg["accepted"] = True
        accepted.append(seg)
    return accepted


def flatten_predictions(frame_idx: int, t: float, preds: List[Dict]) -> tuple[Dict, List[Dict], List[Dict]]:
    """Convert ROI predictions into frame/product/ROI rows.

    The first labels written here are *raw* labels.  Final labels may be updated after temporal post-processing.
    """
    if not preds:
        frame_row = {
            "frame_idx": frame_idx, "time_sec": t, "product_count": 0,
            "raw_ng_product_count": 0, "raw_frame_label": "OK", "frame_label": "OK",
            "max_product_score": 0.0, "max_roi_norm_score": 0.0,
        }
        return frame_row, [], []

    keyed_preds = [(int(p.get("center_x", i)), p) for i, p in enumerate(preds)]
    centers = sorted(set(cx for cx, _p in keyed_preds))
    product_rows: List[Dict] = []
    roi_rows: List[Dict] = []
    raw_ng_count = 0
    max_product_score = 0.0
    for cx in centers:
        ps = [p for pred_cx, p in keyed_preds if pred_cx == cx]
        raw_product_label = "NG" if any(p.get("product_label", p["label"]) == "NG" for p in ps) else "OK"
        product_score = max(float(p.get("product_score", p.get("norm_score", p.get("score", 0.0)))) for p in ps)
        if raw_product_label == "NG":
            raw_ng_count += 1
        max_product_score = max(max_product_score, product_score)
        positions = ";".join(sorted(set(str(p.get("position", "roi")) for p in ps)))
        product_row = {
            "frame_idx": frame_idx, "time_sec": t, "center_x": cx,
            "raw_product_label": raw_product_label, "product_label": raw_product_label,
            "product_score": product_score, "positions": positions, "roi_count": len(ps),
        }
        for key in ("deep_norm_score", "yellow_defect_score", "yellow_ratio", "yellow_threshold", "threshold"):
            vals = [float(p[key]) for p in ps if key in p]
            if vals:
                product_row[key] = max(vals) if key.endswith("_score") or key == "threshold" else float(np.mean(vals))
        product_rows.append(product_row)
        for p in ps:
            x, y, w, h = roi_to_xywh(p.get("roi", [0, 0, 0, 0]))
            raw_roi_label = str(p.get("product_label", p.get("label", "OK"))).upper()
            roi_row = {
                "frame_idx": frame_idx, "time_sec": t, "center_x": cx,
                "position": p.get("position", "roi"), "x": x, "y": y, "w": w, "h": h,
                "raw_roi_label": raw_roi_label, "roi_label": raw_roi_label,
                "norm_score": float(p.get("norm_score", p.get("score", 0.0))),
                "product_score": product_score,
            }
            for key in ("score", "deep_norm_score", "yellow_defect_score", "yellow_ratio", "yellow_threshold", "threshold"):
                if key in p:
                    roi_row[key] = float(p[key])
            if "patch_scores" in p:
                roi_row["patch_scores"] = p["patch_scores"]
            roi_rows.append(roi_row)

    raw_frame_label = "NG" if raw_ng_count > 0 else "OK"
    frame_row = {
        "frame_idx": frame_idx, "time_sec": t, "product_count": len(centers),
        "raw_ng_product_count": raw_ng_count, "ng_product_count": raw_ng_count,
        "raw_frame_label": raw_frame_label, "frame_label": raw_frame_label,
        "max_product_score": max_product_score,
        "max_roi_norm_score": max(float(p.get("norm_score", p.get("score", 0.0))) for p in preds),
    }
    return frame_row, product_rows, roi_rows


def train_model(args) -> object:
    cfg = load_json(args.config) if args.config else {}
    detector = make_detector(args.video_id, cfg)
    summary = detector.train_from_video(args.video, normal_start_sec=args.normal_start,
                                        normal_end_sec=args.normal_end, sample_fps=args.train_fps)
    detector.save(args.model)
    save_json(summary, Path(args.output_dir) / f"video{args.video_id}_train_summary.json")
    return detector


def _expand_segments(segments: List[Dict], *, pre_sec: float = 0.0, post_sec: float = 0.0) -> List[Dict]:
    if not segments:
        return []
    expanded = []
    for seg in segments:
        item = dict(seg)
        item["start_sec"] = max(0.0, float(item["start_sec"]) - max(0.0, float(pre_sec)))
        item["end_sec"] = float(item["end_sec"]) + max(0.0, float(post_sec))
        item["duration_sec"] = max(0.0, float(item["end_sec"]) - float(item["start_sec"]))
        expanded.append(item)
    expanded = sorted(expanded, key=lambda s: float(s["start_sec"]))
    merged: List[Dict] = []
    for seg in expanded:
        if not merged or float(seg["start_sec"]) > float(merged[-1]["end_sec"]):
            merged.append(seg)
        else:
            prev = merged[-1]
            prev["end_sec"] = max(float(prev["end_sec"]), float(seg["end_sec"]))
            prev["duration_sec"] = max(0.0, float(prev["end_sec"]) - float(prev["start_sec"]))
            prev["max_product_score"] = max(float(prev.get("max_product_score", 0.0)), float(seg.get("max_product_score", 0.0)))
            prev["sample_count"] = int(prev.get("sample_count", 0)) + int(seg.get("sample_count", 0))
    return merged


def _apply_final_labels(frame_df: pd.DataFrame, product_df: pd.DataFrame, roi_df: pd.DataFrame,
                        accepted_segments: List[Dict], *, continuous_segment_label: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame_df = frame_df.copy(); product_df = product_df.copy(); roi_df = roi_df.copy()
    if frame_df.empty:
        return frame_df, product_df, roi_df
    accepted_mask = frame_df["time_sec"].apply(lambda t: _is_time_in_segments(float(t), accepted_segments))
    if continuous_segment_label:
        frame_df["frame_label"] = np.where(accepted_mask, "NG", "OK")
    else:
        frame_df["frame_label"] = np.where(accepted_mask & (frame_df["raw_frame_label"] == "NG"), "NG", "OK")

    if not product_df.empty:
        prod_accepted = product_df["time_sec"].apply(lambda t: _is_time_in_segments(float(t), accepted_segments))
        if continuous_segment_label:
            product_df["product_label"] = np.where(prod_accepted, "NG", "OK")
        else:
            product_df["product_label"] = np.where(prod_accepted & (product_df["raw_product_label"] == "NG"), "NG", "OK")
        # recompute final NG product count per frame
        final_ng = product_df.groupby("frame_idx")["product_label"].apply(lambda s: int((s == "NG").sum()))
        frame_df = frame_df.drop(columns=["ng_product_count"], errors="ignore").merge(
            final_ng.rename("ng_product_count"), left_on="frame_idx", right_index=True, how="left")
        frame_df["ng_product_count"] = frame_df["ng_product_count"].fillna(0).astype(int)
        frame_df["frame_label"] = np.where(frame_df["ng_product_count"] > 0, "NG", "OK")
    else:
        frame_df["ng_product_count"] = 0

    if not roi_df.empty:
        roi_accepted = roi_df["time_sec"].apply(lambda t: _is_time_in_segments(float(t), accepted_segments))
        if continuous_segment_label:
            roi_df["roi_label"] = np.where(roi_accepted, "NG", "OK")
        else:
            roi_df["roi_label"] = np.where(roi_accepted & (roi_df["raw_roi_label"] == "NG"), "NG", "OK")
    return frame_df, product_df, roi_df


def _patch_scores_to_heatmap(patch_scores, w: int, h: int, threshold: float | None = None) -> np.ndarray | None:
    if patch_scores is None or w <= 0 or h <= 0:
        return None
    scores = np.asarray(patch_scores, np.float32)
    if scores.ndim == 1:
        side = int(round(np.sqrt(len(scores))))
        if side * side != len(scores):
            return None
        scores = scores.reshape(side, side)
    denom = float(threshold) if threshold and threshold > 1e-9 else float(np.percentile(scores, 99) + 1e-9)
    norm = np.clip(scores / denom, 0.0, 1.0)
    heat = cv2.resize(norm, (int(w), int(h)), interpolation=cv2.INTER_CUBIC)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=2.0)
    return np.clip(heat * 255.0, 0, 255).astype(np.uint8)


def _overlay_heatmap(frame: np.ndarray, row: Dict, alpha: float = 0.42) -> np.ndarray:
    if "patch_scores" not in row:
        return frame
    x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
    H, W = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return frame
    heat = _patch_scores_to_heatmap(row.get("patch_scores"), x1 - x0, y1 - y0, row.get("threshold"))
    if heat is None:
        return frame
    color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    out = frame.copy()
    out[y0:y1, x0:x1] = cv2.addWeighted(out[y0:y1, x0:x1], 1.0 - alpha, color, alpha, 0)
    return out


def _draw_from_roi_rows(frame: np.ndarray, rows: List[Dict], show_heatmap: bool = False) -> np.ndarray:
    out = frame.copy()
    if show_heatmap:
        for r in rows:
            out = _overlay_heatmap(out, r)
    # draw per ROI boxes; label from final post-processed label
    for r in rows:
        x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
        if str(r.get("position", "")).lower() == "face":
            cx, cy = x + w / 2.0, y + h / 2.0
            scale = 1.45
            w, h = int(round(w * scale)), int(round(h * scale))
            x, y = int(round(cx - w / 2.0)), int(round(cy - h / 2.0))
            H, W = frame.shape[:2]
            x, y = max(0, x), max(0, y)
            w, h = min(w, W - x), min(h, H - y)
        lab = str(r.get("roi_label", "OK")).upper()
        color = (0, 205, 0) if lab == "OK" else (0, 0, 255)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
        cv2.putText(out, f"{lab} {float(r.get('product_score', 0.0)):.2f}", (x, max(20, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
    return out


def _render_video_from_roi_rows(args, roi_df: pd.DataFrame, video_out: Path) -> int:
    if roi_df.empty:
        return 0
    out_fps = float(args.video_out_fps or args.infer_fps or 1.0)
    playback_speed = max(0.05, float(getattr(args, "playback_speed", 1.0) or 1.0))

    row_groups = [(float(v["time_sec"].iloc[0]), v.to_dict("records"))
                  for _k, v in roi_df.sort_values("time_sec").groupby("frame_idx")]
    if not row_groups:
        return 0
    group_times = np.asarray([t for t, _rows in row_groups], np.float32)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(args.video)
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_f = max(0, int(round(float(args.test_start) * src_fps)))
    end_f = total if args.test_end is None or float(args.test_end) < 0 else min(total, int(round(float(args.test_end) * src_fps)))
    step = max(1, int(round(src_fps * playback_speed / max(out_fps, 1e-6))))

    writer = None
    written = 0
    for frame_idx in range(start_f, end_f, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        t = frame_idx / src_fps
        nearest = int(np.argmin(np.abs(group_times - t)))
        rows = row_groups[nearest][1]
        detected = _draw_from_roi_rows(frame, rows, show_heatmap=bool(args.render_heatmap_video))
        canvas = side_by_side(frame, detected, "Original", "Detected OK/NG")
        cv2.putText(canvas, f"t={t:.1f}s  normal-train only + temporal postprocess", (20, canvas.shape[0]-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 3, cv2.LINE_AA)
        cv2.putText(canvas, f"t={t:.1f}s  normal-train only + temporal postprocess", (20, canvas.shape[0]-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (45,45,45), 1, cv2.LINE_AA)
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_out), fourcc, out_fps, (canvas.shape[1], canvas.shape[0]))
        writer.write(canvas)
        written += 1
    cap.release()
    if writer is not None:
        writer.release()
    return written


def _export_heatmap_samples(args, roi_df: pd.DataFrame, out_dir: Path) -> int:
    if roi_df.empty or "patch_scores" not in roi_df.columns:
        return 0
    heat_dir = ensure_dir(out_dir / "heatmaps" / f"video{args.video_id}")
    sample_rows = roi_df[roi_df["patch_scores"].notna()].copy()
    if sample_rows.empty:
        return 0
    sample_rows["_rank_score"] = sample_rows.get("product_score", sample_rows.get("norm_score", 0.0)).astype(float)
    sample_rows = sample_rows.sort_values("_rank_score", ascending=False).head(int(args.max_heatmap_samples))

    written = 0
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        return 0
    for i, (_idx, row) in enumerate(sample_rows.iterrows(), 1):
        frame_idx = int(row["frame_idx"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        overlay = _draw_from_roi_rows(frame, [row.to_dict()], show_heatmap=True)
        x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
        H, W = overlay.shape[:2]
        pad = 30
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        crop = overlay[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        name = f"{i:02d}_t{float(row['time_sec']):07.2f}_{str(row.get('position','roi'))}_score{float(row.get('product_score', 0.0)):.2f}.jpg"
        cv2.imwrite(str(heat_dir / name), crop)
        written += 1
    cap.release()
    return written


def run_inference(args, detector) -> Dict:
    out_dir = ensure_dir(args.output_dir)
    frame_csv = out_dir / f"video{args.video_id}_full_test_frame_predictions.csv"
    product_csv = out_dir / f"video{args.video_id}_full_test_product_predictions.csv"
    roi_csv = out_dir / f"video{args.video_id}_full_test_roi_predictions.csv"
    json_path = out_dir / f"video{args.video_id}_full_test_alarm_segments.json"
    video_out = out_dir / f"video{args.video_id}_full_test_side_by_side.mp4"

    frame_rows: List[Dict] = []
    product_rows: List[Dict] = []
    roi_rows: List[Dict] = []

    info = video_info(args.video)
    inference_t0 = time.perf_counter()
    detector_wall_time = 0.0
    for frame_idx, t, frame in iter_video_samples(args.video, args.test_start, args.test_end, args.infer_fps):
        pred_t0 = time.perf_counter()
        preds = detector.predict_frame(frame)
        detector_wall_time += time.perf_counter() - pred_t0
        frame_row, prows, rrows = flatten_predictions(frame_idx, t, preds)
        frame_rows.append(frame_row)
        product_rows.extend(prows)
        roi_rows.extend(rrows)
    inference_wall_time = time.perf_counter() - inference_t0

    frame_df = pd.DataFrame(frame_rows)
    prod_df = pd.DataFrame(product_rows)
    roi_df = pd.DataFrame(roi_rows)

    # Ensure downstream code has expected columns even when a test interval
    # yields no frames or no detected ROI. This makes Video A/B scripts fail
    # gracefully instead of raising KeyError.
    for col, default in {
        "frame_idx": 0, "time_sec": 0.0, "raw_frame_label": "OK",
        "frame_label": "OK", "max_product_score": 0.0,
        "raw_ng_product_count": 0, "ng_product_count": 0,
        "product_count": 0, "max_roi_norm_score": 0.0,
    }.items():
        if col not in frame_df.columns:
            frame_df[col] = pd.Series(dtype=type(default))

    accepted_segments = _accepted_segments_from_frame_df(
        frame_df,
        min_samples=int(args.min_alarm_samples),
        merge_gap_sec=float(args.merge_gap_sec),
        strong_score_threshold=args.segment_strong_score_threshold if args.use_segment_strong_filter else None,
    )
    accepted_segments = _expand_segments(
        accepted_segments,
        pre_sec=float(args.alarm_pre_extend_sec),
        post_sec=float(args.alarm_post_extend_sec),
    )
    frame_df, prod_df, roi_df = _apply_final_labels(
        frame_df, prod_df, roi_df, accepted_segments,
        continuous_segment_label=bool(args.continuous_segment_label),
    )

    # Add final segment stats after label filtering.
    final_flags = (frame_df["frame_label"] == "NG").tolist() if not frame_df.empty else []
    final_segments = group_alarm_segments(frame_df["time_sec"].tolist(), final_flags,
                                          min_samples=int(args.min_alarm_samples),
                                          merge_gap_sec=float(args.merge_gap_sec)) if not frame_df.empty else []
    for seg in final_segments:
        m = (frame_df["time_sec"] >= seg["start_sec"]) & (frame_df["time_sec"] <= seg["end_sec"])
        seg["max_product_score"] = float(frame_df.loc[m, "max_product_score"].max()) if m.any() else 0.0
        seg["ng_sample_count"] = int(m.sum())

    roi_df_for_render = roi_df.copy()
    roi_df_for_csv = roi_df.drop(columns=["patch_scores"], errors="ignore")
    frame_df.to_csv(frame_csv, index=False, encoding="utf-8-sig")
    prod_df.to_csv(product_csv, index=False, encoding="utf-8-sig")
    roi_df_for_csv.to_csv(roi_csv, index=False, encoding="utf-8-sig")

    written = 0
    heatmap_samples = 0
    render_wall_time = 0.0
    if args.render_video:
        render_t0 = time.perf_counter()
        written = _render_video_from_roi_rows(args, roi_df_for_render, video_out)
        render_wall_time = time.perf_counter() - render_t0
    if args.export_heatmaps:
        heatmap_samples = _export_heatmap_samples(args, roi_df_for_render, out_dir)

    raw_ng_count = int((frame_df["raw_frame_label"] == "NG").sum()) if not frame_df.empty else 0
    final_ng_count = int((frame_df["frame_label"] == "NG").sum()) if not frame_df.empty else 0
    sampled_count = int(len(frame_df))
    inference_fps = float(sampled_count / inference_wall_time) if inference_wall_time > 0 else 0.0
    detector_fps = float(sampled_count / detector_wall_time) if detector_wall_time > 0 else 0.0
    summary = {
        "video": str(args.video),
        "video_info": info,
        "test_start": args.test_start,
        "test_end": args.test_end,
        "infer_fps": args.infer_fps,
        "sampled_frames": sampled_count,
        "speed": {
            "inference_wall_time_sec": float(inference_wall_time),
            "detector_wall_time_sec": float(detector_wall_time),
            "render_wall_time_sec": float(render_wall_time),
            "samples_per_sec_including_decode": inference_fps,
            "samples_per_sec_detector_only": detector_fps,
            "sec_per_sample_including_decode": float(inference_wall_time / max(1, sampled_count)),
            "sec_per_sample_detector_only": float(detector_wall_time / max(1, sampled_count)),
        },
        "raw_ng_frame_count": raw_ng_count,
        "raw_ng_rate": float(raw_ng_count / max(1, len(frame_df))),
        "ng_frame_count": final_ng_count,
        "ng_rate": float(final_ng_count / max(1, len(frame_df))),
        "postprocess": {
            "min_alarm_samples": int(args.min_alarm_samples),
            "merge_gap_sec": float(args.merge_gap_sec),
            "use_segment_strong_filter": bool(args.use_segment_strong_filter),
            "segment_strong_score_threshold": args.segment_strong_score_threshold,
            "continuous_segment_label": bool(args.continuous_segment_label),
            "alarm_pre_extend_sec": float(args.alarm_pre_extend_sec),
            "alarm_post_extend_sec": float(args.alarm_post_extend_sec),
            "note": "Manual defect timestamps are not used. Weak candidate alarm segments are accepted only if their own anomaly scores are strong enough.",
        },
        "frame_predictions_csv": str(frame_csv),
        "product_predictions_csv": str(product_csv),
        "roi_predictions_csv": str(roi_csv),
        "rendered_video": str(video_out) if args.render_video else None,
        "rendered_frames": written,
        "heatmap_samples": heatmap_samples,
        "heatmap_dir": str(out_dir / "heatmaps" / f"video{args.video_id}") if args.export_heatmaps else None,
        "alarm_segments": final_segments,
    }
    save_json(summary, json_path)
    save_json(summary, out_dir / f"video{args.video_id}_full_test_summary.json")
    return summary


def main():
    p = argparse.ArgumentParser(description="Train on normal segment and run strict full-test inference.")
    p.add_argument("--video-id", required=True, choices=["A", "B"])
    p.add_argument("--video", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--normal-start", type=float, default=0.0)
    p.add_argument("--normal-end", type=float, default=None)
    p.add_argument("--test-start", type=float, default=None)
    p.add_argument("--test-end", type=float, default=-1.0)
    p.add_argument("--train-fps", type=float, default=None)
    p.add_argument("--infer-fps", type=float, default=None)
    p.add_argument("--video-out-fps", type=float, default=8.0)
    p.add_argument("--playback-speed", type=float, default=None)
    p.add_argument("--min-alarm-samples", type=int, default=None)
    p.add_argument("--merge-gap-sec", type=float, default=None)
    p.add_argument("--segment-strong-score-threshold", type=float, default=None)
    p.add_argument("--disable-segment-strong-filter", action="store_true")
    p.add_argument("--alarm-pre-extend-sec", type=float, default=None)
    p.add_argument("--alarm-post-extend-sec", type=float, default=None)
    p.add_argument("--disable-continuous-segment-label", action="store_true")
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--render-video", action="store_true")
    p.add_argument("--render-heatmap-video", action="store_true")
    p.add_argument("--disable-export-heatmaps", action="store_true")
    p.add_argument("--max-heatmap-samples", type=int, default=None)
    args = p.parse_args()

    cfg = load_json(args.config) if args.config else {}
    if args.normal_end is None:
        args.normal_end = float(cfg.get("normal_end_sec", 360.0 if args.video_id == "B" else 240.0))
    if args.test_start is None:
        args.test_start = float(cfg.get("test_start_sec", args.normal_end))
    if args.test_end is None or args.test_end < 0:
        args.test_end = float(cfg.get("test_end_sec", args.test_end))
    if args.train_fps is None:
        args.train_fps = float(cfg.get("train_sample_fps", 1.0))
    if args.infer_fps is None:
        args.infer_fps = float(cfg.get("infer_sample_fps", 2.0))
    if args.playback_speed is None:
        args.playback_speed = float(cfg.get("playback_speed", 1.0))
    args.playback_speed = max(0.05, float(args.playback_speed))
    args.render_heatmap_video = bool(args.render_heatmap_video or cfg.get("render_heatmap_video", False))
    args.export_heatmaps = bool(cfg.get("export_heatmaps", True)) and not bool(args.disable_export_heatmaps)
    if args.max_heatmap_samples is None:
        args.max_heatmap_samples = int(cfg.get("max_heatmap_samples", 24))
    if args.min_alarm_samples is None:
        args.min_alarm_samples = int(cfg.get("min_alarm_samples", 2))
    if args.merge_gap_sec is None:
        args.merge_gap_sec = float(cfg.get("merge_gap_sec", 3.0))
    if args.segment_strong_score_threshold is None:
        args.segment_strong_score_threshold = cfg.get("segment_strong_score_threshold", 1.0)
    args.use_segment_strong_filter = bool(cfg.get("use_segment_strong_filter", True)) and not bool(args.disable_segment_strong_filter)
    if args.alarm_pre_extend_sec is None:
        args.alarm_pre_extend_sec = float(cfg.get("alarm_pre_extend_sec", 0.0))
    if args.alarm_post_extend_sec is None:
        args.alarm_post_extend_sec = float(cfg.get("alarm_post_extend_sec", 0.0))
    args.continuous_segment_label = bool(cfg.get("continuous_segment_label", True)) and not bool(args.disable_continuous_segment_label)
    if args.model is None:
        args.model = str(Path(args.output_dir) / f"video{args.video_id}_normal_memory_bank.npz")

    ensure_dir(args.output_dir)
    if args.retrain or not Path(args.model).exists():
        detector = train_model(args)
    else:
        detector = load_detector(args.video_id, args.model)
    summary = run_inference(args, detector)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
