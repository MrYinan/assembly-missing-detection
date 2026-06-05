# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
import cv2

from .detectors import make_detector
from .utils import load_json, ensure_dir


def main():
    p = argparse.ArgumentParser(description="Export ROI debug images at selected timestamps. Does not train or predict labels.")
    p.add_argument("--video-id", required=True, choices=["A", "B"])
    p.add_argument("--video", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--times", nargs="+", type=float, required=True, help="timestamps in seconds")
    p.add_argument("--out-dir", default="figures/roi_debug")
    args = p.parse_args()
    cfg = load_json(args.config) if args.config else {}
    det = make_detector(args.video_id, cfg)
    out_dir = ensure_dir(args.out_dir)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(args.video)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    for t in args.times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok:
            continue
        preds = []
        for roi in det.detect_rois(frame) if args.video_id == "A" else det.detect_rois(frame, stable_only=False):
            if args.video_id == "A":
                preds.append({"label":"OK", "roi":roi["circle"], "roi_type":"circle", "norm_score":0.0})
            else:
                preds.append({"label":"OK", "roi":roi["bbox"], "roi_type":"bbox", "position":roi["position"], "center_x":roi["center_x"], "norm_score":0.0})
        img = det.draw(frame, preds, title=f"ROI debug t={t:.1f}s")
        cv2.imwrite(str(out_dir / f"roi_debug_{t:.1f}.jpg"), img)
    cap.release()
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
