# -*- coding: utf-8 -*-
"""
generate_report_figures_final.py

功能：
根据已经运行完成的 Deep PatchCore 推理结果，自动生成实验报告插图。

运行前要求：
1. 已经运行 scripts/run_videoA_deep_patchcore.bat
2. 已经运行 scripts/run_videoB_deep_patchcore.bat
3. outputs/ 下已有对应 CSV / JSON
4. data/ 下有原始视频

本脚本不参与训练、不参与推理、不参与阈值设定；
它只读取结果文件并生成报告图片。
"""

import argparse
import json
import math
from pathlib import Path
import pandas as pd

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_csv(path):
    path = Path(path)
    if not path.exists():
        print(f"[WARN] 缺少文件：{path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json(path):
    path = Path(path)
    if not path.exists():
        print(f"[WARN] 缺少文件：{path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_segments(path):
    data = read_json(path)
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "alarm_segments" in data:
            return data["alarm_segments"]
        if "segments" in data:
            return data["segments"]
    return []


def savefig(path, dpi=180):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[OK] 保存：{path}")


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": fps,
        "frames": frames,
        "width": width,
        "height": height,
        "duration": frames / fps if fps > 0 else 0,
    }


def read_frame(video_path, t):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    idx = int(round(t * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame


def nearest_product_rows(product_df, t, max_dt=0.8):
    if product_df.empty or "time_sec" not in product_df.columns:
        return pd.DataFrame()
    df = product_df.copy()
    df["_dt"] = (df["time_sec"] - t).abs()
    best_dt = df["_dt"].min()
    if best_dt > max_dt:
        return pd.DataFrame()
    best_time = df.loc[df["_dt"].idxmin(), "time_sec"]
    return df[(df["time_sec"] - best_time).abs() < 1e-6].drop(columns=["_dt"])


def get_label(row):
    for col in [
        "roi_label",
        "product_label",
        "label",
        "pred_label",
        "final_label",
        "frame_label",
        "raw_roi_label",
        "raw_product_label",
    ]:
        if col in row.index:
            return str(row[col]).upper()
    return "OK"


def get_score(row):
    for col in ["score", "product_score", "max_product_score", "final_score", "raw_score"]:
        if col in row.index:
            try:
                return float(row[col])
            except Exception:
                pass
    return 0.0


def get_xywh(row):
    """
    更稳的 ROI 解析函数：
    兼容 x/y/w/h、x1/y1/x2/y2、bbox、roi 字符串、圆形 ROI。
    """
    import ast
    import json
    import numpy as np

    # 1. 标准矩形字段
    if all(c in row.index for c in ["x", "y", "w", "h"]):
        try:
            x, y, w, h = row["x"], row["y"], row["w"], row["h"]
            if not any(pd.isna(v) for v in [x, y, w, h]):
                return int(float(x)), int(float(y)), int(float(w)), int(float(h))
        except Exception:
            pass

    # 2. x1/y1/x2/y2 字段
    if all(c in row.index for c in ["x1", "y1", "x2", "y2"]):
        try:
            x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
            if not any(pd.isna(v) for v in [x1, y1, x2, y2]):
                x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
                return int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        except Exception:
            pass

    # 3. 圆形字段
    if all(c in row.index for c in ["cx", "cy", "r"]):
        try:
            cx, cy, r = row["cx"], row["cy"], row["r"]
            if not any(pd.isna(v) for v in [cx, cy, r]):
                cx, cy, r = map(float, [cx, cy, r])
                return int(cx - r), int(cy - r), int(2 * r), int(2 * r)
        except Exception:
            pass

    # 4. roi / bbox 字符串字段，例如 "[x,y,w,h]" 或 "[cx,cy,r]"
    for key in ["roi", "bbox", "box"]:
        if key in row.index:
            val = row[key]
            if pd.isna(val):
                continue

            try:
                if isinstance(val, str):
                    s = val.strip()
                    try:
                        arr = json.loads(s)
                    except Exception:
                        arr = ast.literal_eval(s)
                else:
                    arr = val

                if isinstance(arr, np.ndarray):
                    arr = arr.tolist()

                if isinstance(arr, (list, tuple)):
                    if len(arr) == 4:
                        x, y, w, h = map(float, arr)
                        return int(x), int(y), int(w), int(h)
                    if len(arr) == 3:
                        cx, cy, r = map(float, arr)
                        return int(cx - r), int(cy - r), int(2 * r), int(2 * r)
            except Exception:
                pass

    return 0, 0, 0, 0


def draw_boxes(frame, rows):
    out = frame.copy()
    for _, row in rows.iterrows():
        x, y, w, h = get_xywh(row)
        if w <= 0 or h <= 0:
            continue

        label = get_label(row)
        score = get_score(row)

        color = (0, 0, 255) if label == "NG" else (0, 220, 0)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)

        text = f"{label} {score:.2f}"
        cv2.putText(
            out,
            text,
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def plot_method_pipeline(out_path):
    steps = [
        "工业视频输入",
        "正常段训练\n仅使用 OK 样本",
        "ROI 定位\nA端面 / B接口",
        "ResNet18\n深度特征",
        "PatchCore\nMemory Bank",
        "最近邻距离\n异常分数",
        "时序后处理\n过滤短误报",
        "OK / NG\n可视化输出",
    ]

    fig, ax = plt.subplots(figsize=(16, 2.8))
    ax.axis("off")

    x0 = 0.02
    y0 = 0.35
    w = 0.11
    h = 0.35
    gap = 0.015

    for i, s in enumerate(steps):
        x = x0 + i * (w + gap)
        rect = Rectangle((x, y0), w, h, fill=False, linewidth=1.6)
        ax.add_patch(rect)
        ax.text(x + w / 2, y0 + h / 2, s, ha="center", va="center", fontsize=10)

        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x + w + gap * 0.8, y0 + h / 2),
                xytext=(x + w, y0 + h / 2),
                arrowprops=dict(arrowstyle="->", lw=1.5),
            )

    ax.set_title("ResNet-PatchCore 工业装配件缺失检测流程", fontsize=15)
    savefig(out_path)


def plot_data_split(video_id, duration, normal_start, normal_end, test_start, out_path):
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1)
    ax.set_yticks([])

    unused = []
    if normal_start > 0:
        unused.append((0, normal_start))
    if normal_end < test_start:
        unused.append((normal_end, test_start - normal_end))
    if unused:
        ax.broken_barh(
            unused,
            (0.25, 0.45),
            facecolors="#e6e6e6",
            edgecolors="#b8b8b8",
            label="未使用视频段",
        )

    ax.broken_barh(
        [(normal_start, normal_end - normal_start)],
        (0.25, 0.45),
        facecolors="#2ca25f",
        edgecolors="#17663b",
        label="正常训练段",
    )

    ax.broken_barh(
        [(test_start, duration - test_start)],
        (0.25, 0.45),
        facecolors="#fdae61",
        edgecolors="#b65f00",
        label="测试段自动推理",
    )

    ax.axvline(test_start, linestyle="--", linewidth=1.5, color="#444444")
    ax.text((normal_start + normal_end) / 2, 0.78, f"Train {normal_start:.0f}-{normal_end:.0f}s",
            ha="center", fontsize=10)
    ax.text((test_start + duration) / 2, 0.78, f"Test {test_start:.0f}s-end",
            ha="center", fontsize=10)
    if normal_start > 0:
        ax.text(normal_start / 2, 0.08, "未使用", ha="center", fontsize=9, color="#666666")

    ax.set_xlabel("Time / s")
    ax.set_title(f"Video {video_id} 数据划分")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncols=3, fontsize=9)
    savefig(out_path)


def plot_score_curve(frame_df, segments, out_path, title, clip_score=5.0):
    if frame_df.empty:
        return

    score_col = None
    for col in ["max_product_score", "score", "product_score"]:
        if col in frame_df.columns:
            score_col = col
            break

    if score_col is None:
        print("[WARN] 找不到异常分数字段")
        return

    t = frame_df["time_sec"].values
    y = frame_df[score_col].astype(float).values

    # Video A 会出现 250000 这种黄色缺失强分数，报告图里裁剪显示，避免纵轴炸掉
    y_show = np.clip(y, 0, clip_score)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, y_show, linewidth=1.5, label="Max anomaly score")

    for seg in segments:
        s = float(seg["start_sec"])
        e = float(seg["end_sec"])
        ax.axvspan(s, e, alpha=0.2)

    ax.set_xlabel("Time / s")
    ax.set_ylabel(f"Anomaly score clipped to {clip_score}")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    savefig(out_path)


def plot_alarm_timeline(frame_df, segments, out_path, title):
    if frame_df.empty:
        return

    t_min = float(frame_df["time_sec"].min())
    t_max = float(frame_df["time_sec"].max())

    fig, ax = plt.subplots(figsize=(12, 2.3))
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(0, 1)
    ax.set_yticks([])

    bars = []
    for seg in segments:
        s = float(seg["start_sec"])
        e = float(seg["end_sec"])
        bars.append((s, e - s))

    if bars:
        ax.broken_barh(bars, (0.35, 0.3))

    for seg in segments:
        s = float(seg["start_sec"])
        e = float(seg["end_sec"])
        ax.text((s + e) / 2, 0.72, f"{s:.1f}-{e:.1f}s", ha="center", fontsize=9)

    ax.set_xlabel("Time / s")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    savefig(out_path)


def plot_ng_count(frame_df, out_path, title):
    if frame_df.empty:
        return

    if "ng_count" in frame_df.columns:
        y = frame_df["ng_count"].astype(float).values
    else:
        if "label" not in frame_df.columns:
            return
        y = (frame_df["label"].astype(str).str.upper() == "NG").astype(int).values

    t = frame_df["time_sec"].values

    fig, ax = plt.subplots(figsize=(12, 3.3))
    ax.step(t, y, where="mid")
    ax.set_xlabel("Time / s")
    ax.set_ylabel("NG count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    savefig(out_path)


def plot_alarm_table(segments, out_path, title):
    if not segments:
        return

    rows = []
    for i, seg in enumerate(segments, 1):
        s = float(seg["start_sec"])
        e = float(seg["end_sec"])
        d = float(seg.get("duration_sec", e - s))
        n = int(seg.get("sample_count", 0))
        score = float(seg.get("max_product_score", 0))
        score_show = min(score, 5.0)
        rows.append([i, f"{s:.2f}", f"{e:.2f}", f"{d:.2f}", n, f"{score_show:.3f}"])

    fig, ax = plt.subplots(figsize=(10, max(2.5, len(rows) * 0.5 + 1.5)))
    ax.axis("off")
    ax.set_title(title, fontsize=14)

    table = ax.table(
        cellText=rows,
        colLabels=["#", "Start(s)", "End(s)", "Duration(s)", "Samples", "Max score"],
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    savefig(out_path)


def select_sample_times(frame_df, segments, max_items=6):
    times = []

    # 报警段中点
    for seg in segments:
        s = float(seg["start_sec"])
        e = float(seg["end_sec"])
        times.append((s + e) / 2)

    # 选几个 OK 帧
    if not frame_df.empty:
        if "label" in frame_df.columns:
            ok_df = frame_df[frame_df["label"].astype(str).str.upper() == "OK"]
        else:
            ok_df = frame_df

        if not ok_df.empty:
            idxs = np.linspace(0, len(ok_df) - 1, min(3, len(ok_df))).astype(int)
            for idx in idxs:
                times.append(float(ok_df.iloc[idx]["time_sec"]))

    times = sorted(set(round(t, 2) for t in times))
    return times[:max_items]


def make_detection_montage(video_path, roi_df, sample_times, out_path, title):
    panels = []

    for t in sample_times:
        frame = read_frame(video_path, t)
        if frame is None:
            continue

        rows = nearest_product_rows(roi_df, t)
        det = draw_boxes(frame, rows)

        target_w = 700
        scale = target_w / det.shape[1]
        det = cv2.resize(det, (target_w, int(det.shape[0] * scale)))

        panels.append((t, bgr2rgb(det)))

    if not panels:
        return

    ncols = 2
    nrows = math.ceil(len(panels) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.2 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, (t, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(f"{t:.1f}s 检测结果")
        ax.axis("off")

    fig.suptitle(title, fontsize=16)
    savefig(out_path)


def make_roi_crop_grid(video_path, roi_df, sample_times, out_path, title):
    crops = []

    for t in sample_times:
        frame = read_frame(video_path, t)
        if frame is None:
            continue

        rows = nearest_product_rows(roi_df, t)

        for _, row in rows.iterrows():
            x, y, w, h = get_xywh(row)
            if w <= 0 or h <= 0:
                continue

            H, W = frame.shape[:2]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(W, x + w)
            y2 = min(H, y + h)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            label = get_label(row)
            score = get_score(row)
            crops.append((t, label, score, bgr2rgb(crop)))

            if len(crops) >= 18:
                break

        if len(crops) >= 18:
            break

    if not crops:
        return

    ncols = 3
    nrows = math.ceil(len(crops) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, (t, label, score, crop) in zip(axes, crops):
        ax.imshow(crop)
        ax.set_title(f"{t:.1f}s\n{label} score={score:.2f}", fontsize=10)
        ax.axis("off")

    fig.suptitle(title, fontsize=16)
    savefig(out_path)


def plot_eval_metrics(eval_json, out_path, title):
    data = read_json(eval_json)
    if not isinstance(data, dict):
        return

    metrics = data.get("metrics_on_keyframes_only", {})
    if not metrics:
        return

    keys = ["accuracy", "precision_NG", "recall_NG", "f1_NG"]
    names = ["Accuracy", "Precision", "Recall", "F1"]

    vals = []
    final_names = []
    for k, n in zip(keys, names):
        if k in metrics:
            vals.append(float(metrics[k]))
            final_names.append(n)

    if not vals:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(final_names, vals)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric value")
    ax.set_title(title)

    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")

    savefig(out_path)


def plot_confusion(eval_json, out_path, title):
    data = read_json(eval_json)
    if not isinstance(data, dict) or "counts" not in data:
        return

    c = data["counts"]

    tn = int(c.get("TN", 0))
    fp = int(c.get("FP", 0))
    fn = int(c.get("FN", 0))
    tp = int(c.get("TP", 0))

    mat = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(mat)

    ax.set_xticks([0, 1], labels=["Pred OK", "Pred NG"])
    ax.set_yticks([0, 1], labels=["True OK", "True NG"])
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=18)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(out_path)


def write_summary(video_id, info, frame_df, segments, out_path):
    lines = []
    lines.append(f"# Video {video_id} 自动推理结果摘要\n")
    lines.append(f"- 视频时长：{info['duration']:.2f}s")
    lines.append(f"- 分辨率：{info['width']}×{info['height']}")
    lines.append(f"- FPS：{info['fps']:.3f}")

    if not frame_df.empty:
        lines.append(f"- 测试抽样帧数：{len(frame_df)}")
        if "label" in frame_df.columns:
            ng = int((frame_df["label"].astype(str).str.upper() == "NG").sum())
            lines.append(f"- 最终 NG 帧数：{ng}")
            lines.append(f"- 最终 NG 比例：{ng / max(1, len(frame_df)):.3f}")

    lines.append(f"- 最终报警段数量：{len(segments)}")

    for i, seg in enumerate(segments, 1):
        lines.append(
            f"  - 段 {i}: {float(seg['start_sec']):.2f}s - {float(seg['end_sec']):.2f}s, "
            f"持续 {float(seg.get('duration_sec', 0)):.2f}s, "
            f"max score={float(seg.get('max_product_score', 0)):.3f}"
        )

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] 保存：{out_path}")


def generate_one(video_id, video_path, normal_start, normal_end, test_start, out_dir, clip_score):
    out_dir = Path(out_dir)
    mkdir(out_dir)

    prefix = f"video{video_id}"

    frame_csv = Path("outputs") / f"{prefix}_full_test_frame_predictions.csv"
    roi_csv = Path("outputs") / f"{prefix}_full_test_roi_predictions.csv"
    segments_json = Path("outputs") / f"{prefix}_full_test_alarm_segments.json"
    eval_json = Path("outputs") / f"{prefix}_keyframe_eval.json"

    frame_df = read_csv(frame_csv)
    roi_df = read_csv(roi_csv)
    segments = read_segments(segments_json)
    info = get_video_info(video_path)

    plot_method_pipeline(out_dir / "fig_method_pipeline_deep_patchcore.png")

    plot_data_split(
        video_id,
        info["duration"],
        normal_start,
        normal_end,
        test_start,
        out_dir / f"fig_{prefix}_data_split.png",
    )

    plot_score_curve(
        frame_df,
        segments,
        out_dir / f"fig_{prefix}_full_score_curve.png",
        f"Video {video_id} 完整测试段异常分数曲线",
        clip_score=clip_score,
    )

    plot_alarm_timeline(
        frame_df,
        segments,
        out_dir / f"fig_{prefix}_alarm_timeline.png",
        f"Video {video_id} 最终报警时间轴",
    )

    plot_ng_count(
        frame_df,
        out_dir / f"fig_{prefix}_ng_count_timeline.png",
        f"Video {video_id} 测试段 NG 数量变化",
    )

    plot_alarm_table(
        segments,
        out_dir / f"fig_{prefix}_alarm_segments_table.png",
        f"Video {video_id} 最终报警段统计",
    )

    sample_times = select_sample_times(frame_df, segments)

    make_detection_montage(
        video_path,
        roi_df,
        sample_times,
        out_dir / f"fig_{prefix}_detection_samples.jpg",
        f"Video {video_id} 测试段检测样例",
    )

    make_roi_crop_grid(
        video_path,
        roi_df,
        sample_times,
        out_dir / f"fig_{prefix}_roi_crop_grid.jpg",
        f"Video {video_id} ROI 裁剪样例",
    )

    if eval_json.exists():
        plot_eval_metrics(
            eval_json,
            out_dir / f"fig_{prefix}_keyframe_metrics.png",
            f"Video {video_id} 人工复核关键帧指标",
        )

        plot_confusion(
            eval_json,
            out_dir / f"fig_{prefix}_confusion_matrix.png",
            f"Video {video_id} 人工复核混淆矩阵",
        )

    write_summary(
        video_id,
        info,
        frame_df,
        segments,
        out_dir / f"{prefix}_summary_for_report.md",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video-id", choices=["A", "B"], required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--normal-start", type=float, required=True)
    parser.add_argument("--normal-end", type=float, required=True)
    parser.add_argument("--test-start", type=float, required=True)
    parser.add_argument("--out-dir", default="figures_report")
    parser.add_argument("--clip-score", type=float, default=5.0)

    args = parser.parse_args()

    generate_one(
        video_id=args.video_id,
        video_path=args.video,
        normal_start=args.normal_start,
        normal_end=args.normal_end,
        test_start=args.test_start,
        out_dir=args.out_dir,
        clip_score=args.clip_score,
    )


if __name__ == "__main__":
    main()
