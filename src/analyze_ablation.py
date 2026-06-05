# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .utils import group_alarm_segments, load_json, save_json


def _nearest_keyframe_metrics(frame_df: pd.DataFrame, labels_csv: Path, label_col: str,
                              score_col: str, tolerance: float) -> Dict:
    if not labels_csv.exists() or frame_df.empty or label_col not in frame_df.columns:
        return {}
    labels = pd.read_csv(labels_csv)
    if "time" in labels.columns and "time_sec" not in labels.columns:
        labels = labels.rename(columns={"time": "time_sec"})
    rows = []
    for _, r in labels.iterrows():
        t = float(r["time_sec"])
        true = str(r["label"]).strip().upper()
        idx = (frame_df["time_sec"] - t).abs().idxmin()
        pr = frame_df.loc[idx]
        dt = abs(float(pr["time_sec"]) - t)
        pred = "MISS" if dt > tolerance else str(pr[label_col]).upper()
        rows.append({
            "label_time_sec": t,
            "nearest_time_sec": float(pr["time_sec"]),
            "dt": float(dt),
            "true_label": true,
            "pred_label": pred,
            "score": float(pr.get(score_col, 0.0)),
        })
    eval_df = pd.DataFrame(rows)
    valid = eval_df[eval_df["pred_label"] != "MISS"].copy()
    tp = int(((valid.true_label == "NG") & (valid.pred_label == "NG")).sum())
    tn = int(((valid.true_label == "OK") & (valid.pred_label == "OK")).sum())
    fp = int(((valid.true_label == "OK") & (valid.pred_label == "NG")).sum())
    fn = int(((valid.true_label == "NG") & (valid.pred_label == "OK")).sum())
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "MISS": int((eval_df.pred_label == "MISS").sum()),
        "accuracy": float(acc),
        "precision_NG": float(prec),
        "recall_NG": float(rec),
        "f1_NG": float(f1),
    }


def _summary_row(video_id: str, method: str, frame_df: pd.DataFrame, label_col: str,
                 score_col: str, segments: List[Dict], labels_csv: Path, tolerance: float,
                 raw_ng_baseline: int | None = None, speed: Dict | None = None) -> Dict:
    ng_count = int((frame_df[label_col].astype(str).str.upper() == "NG").sum()) if label_col in frame_df else 0
    sampled = int(len(frame_df))
    row = {
        "video_id": video_id,
        "method": method,
        "sampled_frames": sampled,
        "ng_frame_count": ng_count,
        "ng_rate": float(ng_count / max(1, sampled)),
        "alarm_segment_count": int(len(segments)),
        "alarm_segments": "; ".join(f"{float(s['start_sec']):.1f}-{float(s['end_sec']):.1f}s" for s in segments),
    }
    if raw_ng_baseline is not None:
        row["ng_frame_reduction_vs_raw"] = int(raw_ng_baseline - ng_count)
        row["ng_frame_reduction_rate_vs_raw"] = float((raw_ng_baseline - ng_count) / max(1, raw_ng_baseline))
    if speed:
        row.update({
            "inference_wall_time_sec": float(speed.get("inference_wall_time_sec", 0.0)),
            "samples_per_sec_including_decode": float(speed.get("samples_per_sec_including_decode", 0.0)),
            "sec_per_sample_including_decode": float(speed.get("sec_per_sample_including_decode", 0.0)),
            "samples_per_sec_detector_only": float(speed.get("samples_per_sec_detector_only", 0.0)),
            "sec_per_sample_detector_only": float(speed.get("sec_per_sample_detector_only", 0.0)),
        })
    metrics = _nearest_keyframe_metrics(frame_df, labels_csv, label_col, score_col, tolerance)
    row.update(metrics)
    return row


def _segments_from_labels(frame_df: pd.DataFrame, label_col: str, score_col: str,
                          min_samples: int, merge_gap_sec: float,
                          strong_score_threshold: float | None) -> List[Dict]:
    flags = frame_df[label_col].astype(str).str.upper().eq("NG").tolist()
    segs = group_alarm_segments(frame_df["time_sec"].tolist(), flags, min_samples=min_samples,
                                merge_gap_sec=merge_gap_sec)
    if strong_score_threshold is None:
        return segs
    accepted = []
    for seg in segs:
        m = (frame_df["time_sec"] >= seg["start_sec"]) & (frame_df["time_sec"] <= seg["end_sec"])
        max_score = float(frame_df.loc[m, score_col].max()) if m.any() and score_col in frame_df else 0.0
        if max_score >= float(strong_score_threshold):
            seg = dict(seg)
            seg["max_product_score"] = max_score
            accepted.append(seg)
    return accepted


def _apply_segments(frame_df: pd.DataFrame, raw_col: str, final_col: str, segments: List[Dict]) -> pd.DataFrame:
    out = frame_df.copy()
    if out.empty:
        out[final_col] = "OK"
        return out
    def in_seg(t: float) -> bool:
        return any(float(s["start_sec"]) <= float(t) <= float(s["end_sec"]) for s in segments)
    accepted = out["time_sec"].apply(lambda t: in_seg(float(t)))
    out[final_col] = np.where(accepted & out[raw_col].astype(str).str.upper().eq("NG"), "NG", "OK")
    return out


def _video_a_deep_only_frame_df(frame_df: pd.DataFrame, roi_df: pd.DataFrame, cfg: Dict,
                                post: Dict) -> tuple[pd.DataFrame, List[Dict]]:
    if roi_df.empty or "deep_norm_score" not in roi_df.columns:
        return pd.DataFrame(), []
    threshold = float(cfg.get("product_alert_norm_threshold", 1.0))
    deep_scores = roi_df.groupby("frame_idx")["deep_norm_score"].max().rename("deep_only_score")
    out = frame_df[["frame_idx", "time_sec"]].copy().merge(deep_scores, left_on="frame_idx", right_index=True, how="left")
    out["deep_only_score"] = out["deep_only_score"].fillna(0.0)
    out["deep_only_raw_label"] = np.where(out["deep_only_score"] >= threshold, "NG", "OK")
    min_samples = int(post.get("min_alarm_samples", cfg.get("min_alarm_samples", 2)))
    merge_gap_sec = float(post.get("merge_gap_sec", cfg.get("merge_gap_sec", 3.0)))
    strong = post.get("segment_strong_score_threshold", cfg.get("segment_strong_score_threshold", 1.0))
    if not bool(post.get("use_segment_strong_filter", cfg.get("use_segment_strong_filter", True))):
        strong = None
    segments = _segments_from_labels(out, "deep_only_raw_label", "deep_only_score", min_samples, merge_gap_sec, strong)
    out = _apply_segments(out, "deep_only_raw_label", "deep_only_final_label", segments)
    return out, segments


def analyze_video(video_id: str, args) -> List[Dict]:
    prefix = f"video{video_id}"
    out_dir = Path(args.output_dir)
    cfg = load_json(Path(args.config_dir) / f"{prefix}_deep_patchcore.json")
    summary = load_json(out_dir / f"{prefix}_full_test_summary.json")
    frame_df = pd.read_csv(out_dir / f"{prefix}_full_test_frame_predictions.csv")
    roi_df = pd.read_csv(out_dir / f"{prefix}_full_test_roi_predictions.csv")
    labels_csv = Path(args.labels_dir) / f"{prefix}_keyframes.csv"
    if not labels_csv.exists() and video_id == "B":
        fallback = Path(args.labels_dir) / "videoB_keyframes_full18min.csv"
        if fallback.exists():
            labels_csv = fallback
    post = summary.get("postprocess", {})
    speed = summary.get("speed", {})

    min_samples = int(post.get("min_alarm_samples", cfg.get("min_alarm_samples", 2)))
    merge_gap_sec = float(post.get("merge_gap_sec", cfg.get("merge_gap_sec", 3.0)))
    raw_segments = _segments_from_labels(frame_df, "raw_frame_label", "max_product_score", 1, merge_gap_sec, None)
    final_segments = summary.get("alarm_segments", [])
    raw_ng_count = int((frame_df["raw_frame_label"] == "NG").sum())

    rows = [
        _summary_row(video_id, "raw_candidate_before_temporal_postprocess", frame_df,
                     "raw_frame_label", "max_product_score", raw_segments, labels_csv,
                     args.tolerance, speed=speed),
        _summary_row(video_id, "roi_plus_temporal_postprocess_final", frame_df,
                     "frame_label", "max_product_score", final_segments, labels_csv,
                     args.tolerance, raw_ng_baseline=raw_ng_count, speed=speed),
    ]

    if video_id == "A":
        deep_df, deep_segments = _video_a_deep_only_frame_df(frame_df, roi_df, cfg, post)
        if not deep_df.empty:
            deep_raw_segments = _segments_from_labels(deep_df, "deep_only_raw_label", "deep_only_score",
                                                      1, merge_gap_sec, None)
            deep_raw_ng = int((deep_df["deep_only_raw_label"] == "NG").sum())
            rows.extend([
                _summary_row(video_id, "deep_patchcore_only_raw_candidate_no_yellow_gate", deep_df,
                             "deep_only_raw_label", "deep_only_score", deep_raw_segments, labels_csv,
                             args.tolerance, speed=speed),
                _summary_row(video_id, "deep_patchcore_only_temporal_no_yellow_gate", deep_df,
                             "deep_only_final_label", "deep_only_score", deep_segments, labels_csv,
                             args.tolerance, raw_ng_baseline=deep_raw_ng, speed=speed),
                _summary_row(video_id, "yellow_gate_fusion_plus_temporal_final", frame_df,
                             "frame_label", "max_product_score", final_segments, labels_csv,
                             args.tolerance, raw_ng_baseline=raw_ng_count, speed=speed),
            ])
    return rows


def main():
    p = argparse.ArgumentParser(description="Build raw/final/keyframe/speed ablation tables from existing outputs.")
    p.add_argument("--videos", nargs="+", default=["A", "B"], choices=["A", "B"])
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--labels-dir", default="labels")
    p.add_argument("--out-csv", default="outputs/ablation_summary.csv")
    p.add_argument("--out-json", default="outputs/ablation_summary.json")
    p.add_argument("--out-md", default="outputs/ablation_summary.md")
    p.add_argument("--tolerance", type=float, default=1.0)
    args = p.parse_args()

    rows: List[Dict] = []
    for video_id in args.videos:
        rows.extend(analyze_video(video_id, args))
    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    save_json(rows, args.out_json)
    md_cols = [
        "video_id", "method", "sampled_frames", "ng_frame_count", "ng_rate",
        "alarm_segment_count", "TP", "TN", "FP", "FN", "accuracy",
        "precision_NG", "recall_NG", "f1_NG",
        "ng_frame_reduction_rate_vs_raw",
        "samples_per_sec_including_decode",
    ]
    existing = [c for c in md_cols if c in df.columns]
    md_df = df[existing].copy()
    for col in md_df.columns:
        if pd.api.types.is_float_dtype(md_df[col]):
            md_df[col] = md_df[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    lines = [
        "# Ablation Summary",
        "",
        "Manual keyframe labels are used only for post-inference evaluation.",
        "",
        "| " + " | ".join(existing) + " |",
        "| " + " | ".join(["---"] * len(existing)) + " |",
    ]
    for _, r in md_df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in existing) + " |")
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
