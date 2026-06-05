# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
import json
import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Evaluate already-generated predictions on manual keyframe labels. Labels are NOT used in inference.")
    p.add_argument("--frame-csv", required=True, help="outputs/videoB_full_test_frame_predictions.csv")
    p.add_argument("--labels", required=True, help="CSV with columns time_sec,label or time,label")
    p.add_argument("--out", default="outputs/keyframe_evaluation.json")
    p.add_argument("--tolerance", type=float, default=1.0, help="nearest prediction sample tolerance in seconds")
    args = p.parse_args()

    pred = pd.read_csv(args.frame_csv)
    labels = pd.read_csv(args.labels)
    if "time" in labels.columns and "time_sec" not in labels.columns:
        labels = labels.rename(columns={"time":"time_sec"})
    if "label" not in labels.columns:
        raise ValueError("label CSV needs column: label")
    rows = []
    for _, r in labels.iterrows():
        t = float(r["time_sec"])
        true = str(r["label"]).strip().upper()
        idx = (pred["time_sec"] - t).abs().idxmin()
        pr = pred.loc[idx]
        dt = abs(float(pr["time_sec"]) - t)
        if dt > args.tolerance:
            predicted = "MISS"
        else:
            predicted = str(pr["frame_label"]).upper()
        rows.append({"label_time_sec": t, "nearest_time_sec": float(pr["time_sec"]), "dt": dt,
                     "true_label": true, "pred_label": predicted,
                     "max_product_score": float(pr.get("max_product_score", 0.0))})
    df = pd.DataFrame(rows)
    valid = df[df["pred_label"] != "MISS"].copy()
    tp = int(((valid.true_label == "NG") & (valid.pred_label == "NG")).sum())
    tn = int(((valid.true_label == "OK") & (valid.pred_label == "OK")).sum())
    fp = int(((valid.true_label == "OK") & (valid.pred_label == "NG")).sum())
    fn = int(((valid.true_label == "NG") & (valid.pred_label == "OK")).sum())
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    out = {
        "note": "Manual labels are used only here, after inference.",
        "counts": {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "MISS": int((df.pred_label == "MISS").sum())},
        "metrics_on_keyframes_only": {"accuracy": acc, "precision_NG": prec, "recall_NG": rec, "f1_NG": f1},
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    df.to_csv(str(Path(args.out).with_suffix(".csv")), index=False, encoding="utf-8-sig")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
