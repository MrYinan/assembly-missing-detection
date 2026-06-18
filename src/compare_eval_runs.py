# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .utils import load_json, save_json


def _load_row(run_name: str, output_dir: Path, video_id: str) -> Dict:
    prefix = f"video{video_id}"
    summary = load_json(output_dir / f"{prefix}_full_test_summary.json")
    eval_path = output_dir / f"{prefix}_keyframe_eval.json"
    eval_data = load_json(eval_path) if eval_path.exists() else {}
    counts = eval_data.get("counts", {})
    metrics = eval_data.get("metrics_on_keyframes_only", {})
    speed = summary.get("speed", {})
    return {
        "run": run_name,
        "video_id": video_id,
        "test_start": summary.get("test_start"),
        "test_end": summary.get("test_end"),
        "sampled_frames": summary.get("sampled_frames"),
        "ng_frame_count": summary.get("ng_frame_count"),
        "alarm_segment_count": len(summary.get("alarm_segments", [])),
        "TP": counts.get("TP"),
        "TN": counts.get("TN"),
        "FP": counts.get("FP"),
        "FN": counts.get("FN"),
        "MISS": counts.get("MISS"),
        "accuracy": metrics.get("accuracy"),
        "precision_NG": metrics.get("precision_NG"),
        "recall_NG": metrics.get("recall_NG"),
        "f1_NG": metrics.get("f1_NG"),
        "samples_per_sec_including_decode": speed.get("samples_per_sec_including_decode"),
        "samples_per_sec_detector_only": speed.get("samples_per_sec_detector_only"),
        "sec_per_sample_detector_only": speed.get("sec_per_sample_detector_only"),
    }


def _write_markdown(df: pd.DataFrame, out_md: Path) -> None:
    cols = [
        "run", "video_id", "sampled_frames", "alarm_segment_count",
        "TP", "TN", "FP", "FN", "MISS",
        "accuracy", "precision_NG", "recall_NG", "f1_NG",
        "samples_per_sec_including_decode", "samples_per_sec_detector_only",
    ]
    table = df[[c for c in cols if c in df.columns]].copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    lines = [
        "# Full Evaluation Comparison",
        "",
        "Manual keyframe labels are used only after inference.",
        "",
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join(["---"] * len(table.columns)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in table.columns) + " |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Compare enhanced and baseline full-evaluation runs.")
    p.add_argument("--enhanced-dir", default="outputs_full_eval")
    p.add_argument("--baseline-dir", default="outputs_baseline_eval")
    p.add_argument("--videos", nargs="+", default=["A", "B"], choices=["A", "B"])
    p.add_argument("--out-csv", default="outputs_eval_compare/full_eval_comparison.csv")
    p.add_argument("--out-json", default="outputs_eval_compare/full_eval_comparison.json")
    p.add_argument("--out-md", default="outputs_eval_compare/full_eval_comparison.md")
    args = p.parse_args()

    rows: List[Dict] = []
    for video_id in args.videos:
        rows.append(_load_row("enhanced", Path(args.enhanced_dir), video_id))
        rows.append(_load_row("baseline", Path(args.baseline_dir), video_id))
    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    save_json(rows, args.out_json)
    _write_markdown(df, Path(args.out_md))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
