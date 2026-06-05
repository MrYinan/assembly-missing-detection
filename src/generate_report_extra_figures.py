# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


METHOD_NAMES = {
    "raw_candidate_before_temporal_postprocess": "Raw candidate",
    "roi_plus_temporal_postprocess_final": "ROI + temporal final",
    "deep_patchcore_only_raw_candidate_no_yellow_gate": "Deep-only raw",
    "deep_patchcore_only_temporal_no_yellow_gate": "Deep-only temporal",
    "yellow_gate_fusion_plus_temporal_final": "Yellow fusion final",
}


def _save(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {path}")


def plot_alarm_reduction(df: pd.DataFrame, out_dir: Path):
    rows = df[df["method"].isin([
        "raw_candidate_before_temporal_postprocess",
        "roi_plus_temporal_postprocess_final",
    ])].copy()
    rows["method_label"] = rows["method"].map(METHOD_NAMES)
    videos = ["A", "B"]
    x = range(len(videos))
    width = 0.34
    raw = [int(rows[(rows.video_id == v) & (rows.method == "raw_candidate_before_temporal_postprocess")]["alarm_segment_count"].iloc[0]) for v in videos]
    final = [int(rows[(rows.video_id == v) & (rows.method == "roi_plus_temporal_postprocess_final")]["alarm_segment_count"].iloc[0]) for v in videos]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar([i - width / 2 for i in x], raw, width, label="Raw candidate")
    ax.bar([i + width / 2 for i in x], final, width, label="Final alarm")
    ax.set_xticks(list(x), [f"Video {v}" for v in videos])
    ax.set_ylabel("Alarm segment count")
    ax.set_title("Temporal post-processing reduces fragmented alarms")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    for i, (r, f) in enumerate(zip(raw, final)):
        ax.text(i - width / 2, r + 0.2, str(r), ha="center")
        ax.text(i + width / 2, f + 0.2, str(f), ha="center")
    _save(out_dir / "fig_ablation_alarm_reduction.png")


def plot_video_a_ablation(df: pd.DataFrame, out_dir: Path):
    rows = df[df["video_id"].eq("A") & df["method"].isin([
        "deep_patchcore_only_temporal_no_yellow_gate",
        "yellow_gate_fusion_plus_temporal_final",
        "roi_plus_temporal_postprocess_final",
    ])].copy()
    rows["method_label"] = rows["method"].map(METHOD_NAMES)
    order = [
        "Deep-only temporal",
        "Yellow fusion final",
        "ROI + temporal final",
    ]
    rows["method_label"] = pd.Categorical(rows["method_label"], categories=order, ordered=True)
    rows = rows.sort_values("method_label")
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = range(len(rows))
    ax.bar(x, rows["recall_NG"].astype(float), label="Recall NG")
    ax.plot(x, rows["f1_NG"].astype(float), marker="o", linewidth=2, label="F1 NG")
    ax.set_xticks(list(x), rows["method_label"], rotation=12, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Metric value")
    ax.set_title("Video A ablation: yellow cue improves missing-liner recall")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    for i, v in enumerate(rows["recall_NG"].astype(float)):
        ax.text(i, v + 0.025, f"{v:.2f}", ha="center")
    _save(out_dir / "fig_ablation_videoA_yellow_gate.png")


def plot_speed(df: pd.DataFrame, out_dir: Path):
    rows = df[df["method"].eq("roi_plus_temporal_postprocess_final")].copy()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = [f"Video {v}" for v in rows["video_id"]]
    vals = rows["samples_per_sec_including_decode"].astype(float).tolist()
    ax.bar(labels, vals)
    ax.set_ylabel("Samples per second")
    ax.set_title("Inference throughput on sampled frames")
    ax.grid(True, axis="y", alpha=0.25)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.03, f"{v:.3f}", ha="center")
    _save(out_dir / "fig_speed_samples_per_second.png")


def main():
    df = pd.read_csv("outputs/ablation_summary.csv")
    out_dir = Path("figures_report")
    plot_alarm_reduction(df, out_dir)
    plot_video_a_ablation(df, out_dir)
    plot_speed(df, out_dir)


if __name__ == "__main__":
    main()
