# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description="Generate full-test plots from prediction CSVs.")
    p.add_argument("--frame-csv", required=True)
    p.add_argument("--segments-json", default=None)
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--prefix", default="videoB")
    args = p.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.frame_csv)
    if df.empty:
        raise RuntimeError("Empty frame CSV")

    raw_col = "raw_frame_label" if "raw_frame_label" in df.columns else "frame_label"
    final_col = "frame_label"

    # 1. max anomaly score curve + final NG markers
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["time_sec"], df["max_product_score"], linewidth=1.5, label="max product score")
    ng = df[final_col].astype(str).str.upper().eq("NG")
    if ng.any():
        ax.scatter(df.loc[ng, "time_sec"], df.loc[ng, "max_product_score"], s=18, label="final NG")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Max product normalized anomaly score")
    ax.set_title("Full test segment anomaly score curve")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / f"{args.prefix}_full_score_curve.png", dpi=180)
    plt.close(fig)

    # 2. raw vs final alarm timeline
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.scatter(df["time_sec"], df[raw_col].astype(str).str.upper().eq("NG").astype(int), s=10, alpha=0.45, label="raw candidate")
    ax.scatter(df["time_sec"], df[final_col].astype(str).str.upper().eq("NG").astype(int) + 0.05, s=13, label="final alarm")
    ax.set_xlabel("Time (s)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["OK", "NG"])
    ax.set_title("Automatic alarm timeline: raw candidates vs final alarms")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / f"{args.prefix}_alarm_timeline.png", dpi=180)
    plt.close(fig)

    # 3. NG count timeline
    count_col = "ng_product_count" if "ng_product_count" in df.columns else None
    fig, ax = plt.subplots(figsize=(10, 3.2))
    if count_col:
        ax.step(df["time_sec"], df[count_col], where="mid", label="final NG product count")
    if "raw_ng_product_count" in df.columns:
        ax.step(df["time_sec"], df["raw_ng_product_count"], where="mid", alpha=0.45, label="raw candidate count")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("NG product count")
    ax.set_title("Frame-level NG count timeline")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / f"{args.prefix}_ng_count_timeline.png", dpi=180)
    plt.close(fig)

    # 4. segment table as figure
    if args.segments_json and Path(args.segments_json).exists():
        with open(args.segments_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        segs = data.get("alarm_segments", []) if isinstance(data, dict) else data
        rows = []
        for i, s in enumerate(segs, 1):
            rows.append([i, f"{s['start_sec']:.1f}", f"{s['end_sec']:.1f}", f"{s['duration_sec']:.1f}", f"{s.get('max_product_score', 0):.2f}", f"{s.get('sample_count', 0)}"])
        fig, ax = plt.subplots(figsize=(9, max(2.0, 0.35 * len(rows) + 1.2)))
        ax.axis("off")
        table = ax.table(cellText=rows if rows else [["-", "-", "-", "-", "-", "-"]],
                         colLabels=["#", "Start(s)", "End(s)", "Duration(s)", "Max score", "Samples"],
                         loc="center")
        table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.35)
        ax.set_title("Final automatic alarm segments")
        fig.tight_layout()
        fig.savefig(out / f"{args.prefix}_alarm_segments_table.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
