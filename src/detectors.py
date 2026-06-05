# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import cv2
import numpy as np

from .deep_patchcore import DeepBackboneConfig, TorchPatchFeatureExtractor, DeepPatchMemoryBank
from .roi_locators import VideoACircleRoiLocator, VideoARoiConfig, VideoBSidePortRoiLocator, VideoBRoiConfig
from .utils import iter_video_samples


@dataclass
class DeepDetectorConfig:
    # Training / inference timing is passed by CLI, but defaults are kept here.
    train_sample_fps: float = 0.5
    infer_sample_fps: float = 1.0
    max_memory_patches: int = 25000
    threshold_quantile: float = 0.99
    threshold_margin: float = 1.05
    top_percent: float = 0.01
    min_validation_count: int = 20
    product_alert_norm_threshold: float = 1.0
    min_laplacian_var: float = 5.0
    seed: int = 42
    # Deep backbone
    arch: str = 'resnet18'
    pretrained: bool = True
    input_size: int = 224
    layers: List[str] = field(default_factory=lambda: ['layer2', 'layer3'])
    device: str = 'auto'
    batch_size: int = 16
    allow_untrained_fallback: bool = False
    # Optional Video A color sanity check. This is not the main model; it avoids
    # obvious non-yellow misses in the circular-liner video.
    # The threshold is learned from the normal training segment only. During
    # inference the color deviation is fused with the Deep PatchCore memory
    # distance as an additional anomaly cue; it is not a hand-coded test-time
    # timestamp rule.
    videoA_use_yellow_gate: bool = True
    videoA_yellow_gate_quantile: float = 0.01
    videoA_yellow_gate_margin: float = 0.75
    videoA_min_yellow_floor: float = 0.18
    videoA_fuse_yellow_score: bool = True


class BaseDeepPatchCoreDetector:
    model_type = 'deep_patchcore_base'

    def __init__(self, cfg: DeepDetectorConfig):
        self.cfg = cfg
        self.extractor: Optional[TorchPatchFeatureExtractor] = None
        self.train_summary: Dict = {}

    def _make_extractor(self) -> TorchPatchFeatureExtractor:
        if self.extractor is None:
            bcfg = DeepBackboneConfig(
                arch=self.cfg.arch,
                pretrained=bool(self.cfg.pretrained),
                layers=tuple(self.cfg.layers),
                input_size=int(self.cfg.input_size),
                device=self.cfg.device,
                batch_size=int(self.cfg.batch_size),
                allow_untrained_fallback=bool(self.cfg.allow_untrained_fallback),
            )
            self.extractor = TorchPatchFeatureExtractor(bcfg)
        return self.extractor

    @staticmethod
    def crop_quality(crop: np.ndarray) -> Dict[str, float]:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return {'lap_var': float(cv2.Laplacian(gray, cv2.CV_64F).var()), 'mean': float(gray.mean()), 'std': float(gray.std())}

    def _crop_bbox(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        x, y, w, h = map(int, bbox)
        H, W = frame.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        return frame[y0:y1, x0:x1]

    def _crop_circle(self, frame: np.ndarray, circle: List[int]) -> np.ndarray:
        x, y, r = map(int, circle)
        H, W = frame.shape[:2]
        pad = int(r * 0.9)
        return frame[max(0, y - pad):min(H, y + pad), max(0, x - pad):min(W, x + pad)]


class VideoBDeepPatchCoreDetector(BaseDeepPatchCoreDetector):
    model_type = 'videoB_resnet_patchcore_side_port_v15'

    def __init__(self, cfg: Optional[Dict] = None):
        cfg = cfg or {}
        deep_keys = set(DeepDetectorConfig.__dataclass_fields__.keys())
        roi_keys = set(VideoBRoiConfig.__dataclass_fields__.keys())
        self.deep_cfg = DeepDetectorConfig(**{k: v for k, v in cfg.items() if k in deep_keys})
        super().__init__(self.deep_cfg)
        self.roi_cfg = VideoBRoiConfig(**{k: v for k, v in cfg.items() if k in roi_keys})
        self.locator = VideoBSidePortRoiLocator(self.roi_cfg)
        self.banks = {
            'top': DeepPatchMemoryBank(self.cfg.max_memory_patches, self.cfg.seed),
            'bottom': DeepPatchMemoryBank(self.cfg.max_memory_patches, self.cfg.seed + 1),
        }

    def detect_rois(self, frame: np.ndarray, stable_only: bool = True) -> List[Dict]:
        return self.locator.detect_rois(frame, stable_only=stable_only)

    def _roi_crops(self, frame: np.ndarray) -> List[Tuple[Dict, np.ndarray, Dict[str, float]]]:
        out = []
        for roi in self.locator.detect_rois(frame, stable_only=True):
            crop = self._crop_bbox(frame, roi['bbox'])
            if crop.size == 0:
                continue
            q = self.crop_quality(crop)
            out.append((roi, crop, q))
        return out

    def train_from_video(self, video_path: str | Path, normal_start_sec: float = 0.0,
                         normal_end_sec: Optional[float] = None, sample_fps: Optional[float] = None) -> Dict:
        if normal_end_sec is None or normal_end_sec <= normal_start_sec:
            raise ValueError('normal_end_sec must be greater than normal_start_sec.')
        sample_fps = float(sample_fps or self.cfg.train_sample_fps)
        extractor = self._make_extractor()
        roi_sets = {'top': [], 'bottom': []}
        sampled_frames = 0; detected = 0; skipped_quality = 0
        for _idx, _t, frame in iter_video_samples(video_path, normal_start_sec, normal_end_sec, sample_fps):
            sampled_frames += 1
            triples = self._roi_crops(frame)
            if not triples:
                continue
            crops = []
            roi_meta = []
            for roi, crop, q in triples:
                if q['lap_var'] < self.cfg.min_laplacian_var:
                    skipped_quality += 1
                    continue
                crops.append(crop); roi_meta.append(roi)
            feats = extractor.extract_batch(crops)
            for roi, pf in zip(roi_meta, feats):
                roi_sets[roi['position']].append(pf)
                detected += 1
        summary = {
            'model_type': self.model_type,
            'video_path': str(video_path),
            'normal_start_sec': float(normal_start_sec),
            'normal_end_sec': float(normal_end_sec),
            'train_sample_fps': sample_fps,
            'sampled_frames': sampled_frames,
            'detected_normal_rois': detected,
            'skipped_by_quality': skipped_quality,
            'backbone': asdict(self.cfg),
            'roi_config': asdict(self.roi_cfg),
            'banks': {},
        }
        for pos in ('top', 'bottom'):
            if len(roi_sets[pos]) < 10:
                raise RuntimeError(f'Too few normal {pos} ROI feature sets: {len(roi_sets[pos])}. Check ROI config or train segment.')
            summary['banks'][pos] = self.banks[pos].fit(
                roi_sets[pos], threshold_quantile=self.cfg.threshold_quantile,
                threshold_margin=self.cfg.threshold_margin, top_percent=self.cfg.top_percent,
                min_validation_count=self.cfg.min_validation_count,
            )
        self.train_summary = summary
        return summary

    def predict_frame(self, frame: np.ndarray) -> List[Dict]:
        extractor = self._make_extractor()
        triples = self._roi_crops(frame)
        crops = [c for _, c, _ in triples]
        feats = extractor.extract_batch(crops) if crops else []
        preds: List[Dict] = []
        by_center: Dict[int, List[Dict]] = {}
        for (roi, _crop, q), pf in zip(triples, feats):
            pos = roi['position']
            score = self.banks[pos].score_roi(pf, top_percent=self.cfg.top_percent)
            thr = float(self.banks[pos].threshold)
            norm = float(score / (thr + 1e-9))
            p = {
                'label': 'NG' if norm >= 1.0 else 'OK',
                'score': float(score),
                'norm_score': norm,
                'threshold': thr,
                'position': pos,
                'center_x': int(roi['center_x']),
                'roi': roi['bbox'],
                'roi_type': 'bbox',
                'lap_var': q['lap_var'],
            }
            preds.append(p)
            by_center.setdefault(int(roi['center_x']), []).append(p)
        alert_thr = float(self.cfg.product_alert_norm_threshold)
        for cx, ps in by_center.items():
            product_score = max(p['norm_score'] for p in ps)
            product_label = 'NG' if product_score >= alert_thr else 'OK'
            for p in ps:
                p['product_score'] = float(product_score)
                p['product_label'] = product_label
                p['label'] = product_label
                p['product_alert_norm_threshold'] = alert_thr
        return preds

    def draw(self, frame: np.ndarray, preds: List[Dict], title='Deep ResNet-PatchCore') -> np.ndarray:
        out = frame.copy()
        cv2.putText(out, title, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255,255,255), 3, cv2.LINE_AA)
        cv2.putText(out, title, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (35,35,35), 1, cv2.LINE_AA)
        for p in preds:
            x, y, w, h = map(int, p['roi'])
            label = p.get('product_label', p['label'])
            color = (0, 205, 0) if label == 'OK' else (0, 0, 255)
            cv2.rectangle(out, (x, y), (x+w, y+h), color, 2, cv2.LINE_AA)
            cv2.putText(out, f'{label} {p.get("product_score", p["norm_score"]):.2f}', (x, max(20, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        d = {'model_type': self.model_type,
             'deep_cfg': json.dumps(asdict(self.cfg), ensure_ascii=False),
             'roi_cfg': json.dumps(asdict(self.roi_cfg), ensure_ascii=False),
             'train_summary': json.dumps(self.train_summary, ensure_ascii=False)}
        for pos, bank in self.banks.items():
            d.update(bank.to_npz_dict(pos))
        np.savez_compressed(path, **d)

    @classmethod
    def load(cls, path: str | Path) -> 'VideoBDeepPatchCoreDetector':
        data = np.load(path, allow_pickle=True)
        deep_cfg = json.loads(str(data['deep_cfg']))
        roi_cfg = json.loads(str(data['roi_cfg']))
        det = cls({**deep_cfg, **roi_cfg})
        for pos in ('top', 'bottom'):
            det.banks[pos] = DeepPatchMemoryBank.from_npz_dict(data, pos, det.cfg.max_memory_patches, det.cfg.seed)
        det.train_summary = json.loads(str(data['train_summary']))
        return det


class VideoADeepPatchCoreDetector(BaseDeepPatchCoreDetector):
    model_type = 'videoA_resnet_patchcore_circle_v15'

    def __init__(self, cfg: Optional[Dict] = None):
        cfg = cfg or {}
        deep_keys = set(DeepDetectorConfig.__dataclass_fields__.keys())
        roi_keys = set(VideoARoiConfig.__dataclass_fields__.keys())
        self.deep_cfg = DeepDetectorConfig(**{k: v for k, v in cfg.items() if k in deep_keys})
        super().__init__(self.deep_cfg)
        self.roi_cfg = VideoARoiConfig(**{k: v for k, v in cfg.items() if k in roi_keys})
        self.locator = VideoACircleRoiLocator(self.roi_cfg)
        self.bank = DeepPatchMemoryBank(self.cfg.max_memory_patches, self.cfg.seed)
        self.yellow_threshold = 0.0

    def detect_rois(self, frame: np.ndarray, stable_only: bool = True) -> List[Dict]:
        return self.locator.detect_rois(frame, stable_only=stable_only)

    def _roi_crops(self, frame: np.ndarray) -> List[Tuple[Dict, np.ndarray, float]]:
        out = []
        for roi in self.locator.detect_rois(frame, stable_only=True):
            crop = self._crop_circle(frame, roi['circle'])
            if crop.size == 0:
                continue
            yr = self.locator.yellow_ratio(crop)
            out.append((roi, crop, yr))
        return out

    def train_from_video(self, video_path: str | Path, normal_start_sec: float = 0.0,
                         normal_end_sec: Optional[float] = None, sample_fps: Optional[float] = None) -> Dict:
        if normal_end_sec is None or normal_end_sec <= normal_start_sec:
            raise ValueError('normal_end_sec must be greater than normal_start_sec.')
        sample_fps = float(sample_fps or self.cfg.train_sample_fps)
        extractor = self._make_extractor()
        roi_features: List[np.ndarray] = []
        yellow: List[float] = []
        sampled = 0; detected = 0
        for _idx, _t, frame in iter_video_samples(video_path, normal_start_sec, normal_end_sec, sample_fps):
            sampled += 1
            triples = self._roi_crops(frame)
            crops = [c for _, c, _ in triples]
            feats = extractor.extract_batch(crops) if crops else []
            for (_roi, _crop, yr), pf in zip(triples, feats):
                roi_features.append(pf); yellow.append(yr); detected += 1
        if len(roi_features) < 10:
            raise RuntimeError(f'Too few normal Video A ROI feature sets: {len(roi_features)}')
        bank_info = self.bank.fit(roi_features, self.cfg.threshold_quantile, self.cfg.threshold_margin, self.cfg.top_percent, self.cfg.min_validation_count)
        y = np.asarray(yellow, np.float32)
        if self.cfg.videoA_use_yellow_gate:
            self.yellow_threshold = max(self.cfg.videoA_min_yellow_floor, float(np.quantile(y, self.cfg.videoA_yellow_gate_quantile) * self.cfg.videoA_yellow_gate_margin))
        else:
            self.yellow_threshold = -1.0
        self.train_summary = {
            'model_type': self.model_type,
            'video_path': str(video_path),
            'normal_start_sec': float(normal_start_sec),
            'normal_end_sec': float(normal_end_sec),
            'train_sample_fps': sample_fps,
            'sampled_frames': sampled,
            'detected_normal_rois': detected,
            'backbone': asdict(self.cfg),
            'roi_config': asdict(self.roi_cfg),
            'bank': bank_info,
            'yellow_threshold': float(self.yellow_threshold),
        }
        return self.train_summary

    def predict_frame(self, frame: np.ndarray) -> List[Dict]:
        extractor = self._make_extractor()
        triples = self._roi_crops(frame)
        feats = extractor.extract_batch([c for _, c, _ in triples]) if triples else []
        preds = []
        for (roi, _crop, yr), pf in zip(triples, feats):
            score = self.bank.score_roi(pf, self.cfg.top_percent)
            deep_norm = float(score / (self.bank.threshold + 1e-9))

            # Video A has a very strong semantic cue: a normal product has a
            # yellow liner on the circular end face, while missing/unremoved
            # cases are pale/transparent.  Earlier versions only used this cue
            # to set the raw label, but the temporal post-process filtered those
            # detections because their Deep PatchCore score could be below 1.0.
            # Here the color deviation is converted to a normalised anomaly
            # score learned from the normal training segment and fused with the
            # deep memory-bank score.  This keeps the method non-leaking: no
            # test timestamp or defect sample is used.
            if self.cfg.videoA_use_yellow_gate and self.yellow_threshold > 0:
                yellow_defect_score = float(self.yellow_threshold / max(float(yr), 1e-6))
            else:
                yellow_defect_score = 0.0

            if getattr(self.cfg, 'videoA_fuse_yellow_score', True):
                product_score = max(deep_norm, yellow_defect_score)
            else:
                product_score = deep_norm

            label = 'NG' if product_score >= self.cfg.product_alert_norm_threshold else 'OK'
            x, y, r = roi['circle']
            preds.append({
                'label': label,
                'score': float(score),
                'norm_score': float(deep_norm),
                'deep_norm_score': float(deep_norm),
                'yellow_defect_score': float(yellow_defect_score),
                'threshold': float(self.bank.threshold),
                'yellow_ratio': float(yr),
                'yellow_threshold': float(self.yellow_threshold),
                'roi': [x, y, r],
                'roi_type': 'circle',
                'center_x': int(x),
                'position': 'face',
                'product_score': float(product_score),
            })
        return preds

    def draw(self, frame: np.ndarray, preds: List[Dict], title='Deep ResNet-PatchCore') -> np.ndarray:
        out = frame.copy()
        cv2.putText(out, title, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (255,255,255), 3, cv2.LINE_AA)
        cv2.putText(out, title, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (35,35,35), 1, cv2.LINE_AA)
        for p in preds:
            x, y, r = map(int, p['roi'])
            color = (0,205,0) if p['label'] == 'OK' else (0,0,255)
            cv2.circle(out, (x, y), r, color, 2, cv2.LINE_AA)
            cv2.putText(out, f'{p["label"]} {p["norm_score"]:.2f}', (x-r, max(20, y-r-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        d = {'model_type': self.model_type,
             'deep_cfg': json.dumps(asdict(self.cfg), ensure_ascii=False),
             'roi_cfg': json.dumps(asdict(self.roi_cfg), ensure_ascii=False),
             'train_summary': json.dumps(self.train_summary, ensure_ascii=False),
             'yellow_threshold': np.asarray([self.yellow_threshold], np.float32)}
        d.update(self.bank.to_npz_dict('bank'))
        np.savez_compressed(path, **d)

    @classmethod
    def load(cls, path: str | Path) -> 'VideoADeepPatchCoreDetector':
        data = np.load(path, allow_pickle=True)
        deep_cfg = json.loads(str(data['deep_cfg']))
        roi_cfg = json.loads(str(data['roi_cfg']))
        det = cls({**deep_cfg, **roi_cfg})
        det.bank = DeepPatchMemoryBank.from_npz_dict(data, 'bank', det.cfg.max_memory_patches, det.cfg.seed)
        det.yellow_threshold = float(data['yellow_threshold'][0])
        det.train_summary = json.loads(str(data['train_summary']))
        return det


def make_detector(video_id: str, cfg: Optional[Dict] = None):
    vid = video_id.upper()
    if vid == 'A':
        return VideoADeepPatchCoreDetector(cfg)
    if vid == 'B':
        return VideoBDeepPatchCoreDetector(cfg)
    raise ValueError('video_id must be A or B')


def load_detector(video_id: str, model_path: str | Path):
    vid = video_id.upper()
    if vid == 'A':
        return VideoADeepPatchCoreDetector.load(model_path)
    if vid == 'B':
        return VideoBDeepPatchCoreDetector.load(model_path)
    raise ValueError('video_id must be A or B')
