# -*- coding: utf-8 -*-
"""
Minimal Deep PatchCore implementation for this assignment.

It follows the key idea of PatchCore:
1) Use a pre-trained deep backbone to extract patch-level features from normal ROIs.
2) Build a normal memory bank.
3) At test time, score each ROI by nearest-neighbour distance to the memory bank.

This is not a YOLO detector and does not use defect frames in training.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import json

import cv2
import numpy as np


@dataclass
class DeepBackboneConfig:
    arch: str = 'resnet18'              # resnet18 / resnet50 / wide_resnet50_2 if torchvision supports it
    pretrained: bool = True
    layers: Tuple[str, ...] = ('layer2', 'layer3')
    input_size: int = 224
    device: str = 'auto'               # auto/cpu/cuda
    batch_size: int = 16
    l2_normalize: bool = True
    allow_untrained_fallback: bool = False
    patch_neighborhood_size: int = 1       # 1 keeps the baseline; 3 enables local PatchCore-style aggregation.


class TorchPatchFeatureExtractor:
    """Extract patch embeddings from ResNet intermediate feature maps.

    Importing torch/torchvision is delayed to runtime so scripts that only inspect
    configs do not fail if the deep-learning environment has not been installed yet.
    """
    def __init__(self, cfg: DeepBackboneConfig):
        self.cfg = cfg
        if int(self.cfg.patch_neighborhood_size) < 1 or int(self.cfg.patch_neighborhood_size) % 2 == 0:
            raise ValueError('patch_neighborhood_size must be a positive odd integer, e.g. 1 or 3.')
        self._load_torch()
        self.model, self.feature_dim = self._load_model()
        self.activations: Dict[str, object] = {}
        self._register_hooks()

    def _load_torch(self):
        try:
            import torch
            import torch.nn.functional as F
            self.torch = torch
            self.F = F
        except Exception as e:
            raise RuntimeError('PyTorch is required for the deep PatchCore version. Install torch first.') from e
        if self.cfg.device == 'auto':
            self.device = 'cuda' if self.torch.cuda.is_available() else 'cpu'
        else:
            self.device = self.cfg.device

    def _load_model(self):
        try:
            import torchvision.models as models
        except Exception as e:
            raise RuntimeError(
                'torchvision is required for ResNet-PatchCore. '\
                'If torchvision import fails, reinstall matching torch/torchvision versions.'
            ) from e

        arch = self.cfg.arch.lower()
        weights = None
        model_fn = None
        try:
            if arch == 'resnet18':
                model_fn = models.resnet18
                weights = models.ResNet18_Weights.DEFAULT if self.cfg.pretrained else None
            elif arch == 'resnet50':
                model_fn = models.resnet50
                weights = models.ResNet50_Weights.DEFAULT if self.cfg.pretrained else None
            elif arch == 'wide_resnet50_2':
                model_fn = models.wide_resnet50_2
                weights = models.Wide_ResNet50_2_Weights.DEFAULT if self.cfg.pretrained else None
            else:
                raise ValueError(f'Unsupported backbone arch: {self.cfg.arch}')
            model = model_fn(weights=weights)
        except Exception as e:
            if not self.cfg.allow_untrained_fallback:
                raise RuntimeError(
                    'Failed to load pretrained torchvision backbone. '\
                    'Connect to the internet once so torchvision can download weights, '\
                    'or set allow_untrained_fallback=true for code testing only.'
                ) from e
            # old torchvision compatibility / offline code testing fallback
            try:
                model = model_fn(pretrained=False)
            except TypeError:
                model = model_fn(weights=None)

        model.eval().to(self.device)
        for p in model.parameters():
            p.requires_grad_(False)
        return model, None

    def _register_hooks(self):
        modules = dict(self.model.named_modules())
        for lname in self.cfg.layers:
            if lname not in modules:
                raise ValueError(f'Layer {lname} not found in backbone. Available examples: layer1/layer2/layer3/layer4')
            def hook(_module, _inp, out, name=lname):
                self.activations[name] = out.detach()
            modules[lname].register_forward_hook(hook)

    def _preprocess_batch(self, crops: Sequence[np.ndarray]):
        arrs = []
        for crop in crops:
            if crop is None or crop.size == 0:
                crop = np.zeros((self.cfg.input_size, self.cfg.input_size, 3), np.uint8)
            crop = cv2.resize(crop, (self.cfg.input_size, self.cfg.input_size), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
            arrs.append(np.transpose(rgb, (2, 0, 1)))
        x = self.torch.from_numpy(np.stack(arrs).astype(np.float32)).to(self.device)
        return x

    def extract_batch(self, crops: Sequence[np.ndarray]) -> List[np.ndarray]:
        if not crops:
            return []
        results: List[np.ndarray] = []
        bs = max(1, int(self.cfg.batch_size))
        for i in range(0, len(crops), bs):
            batch = crops[i:i + bs]
            x = self._preprocess_batch(batch)
            self.activations.clear()
            with self.torch.no_grad():
                _ = self.model(x)
            feats = [self.activations[name] for name in self.cfg.layers]
            k = int(self.cfg.patch_neighborhood_size)
            if k > 1:
                feats = [self.F.avg_pool2d(f, kernel_size=k, stride=1, padding=k // 2) for f in feats]
            ref_h, ref_w = feats[0].shape[-2], feats[0].shape[-1]
            ups = []
            for f in feats:
                if f.shape[-2:] != (ref_h, ref_w):
                    f = self.F.interpolate(f, size=(ref_h, ref_w), mode='bilinear', align_corners=False)
                ups.append(f)
            fcat = self.torch.cat(ups, dim=1)  # B,C,H,W
            fcat = fcat.permute(0, 2, 3, 1).contiguous()  # B,H,W,C
            if self.cfg.l2_normalize:
                fcat = self.F.normalize(fcat, p=2, dim=-1)
            f_np = fcat.cpu().numpy().astype(np.float32)
            for j in range(f_np.shape[0]):
                results.append(f_np[j].reshape(-1, f_np.shape[-1]))
        return results

    def extract_one(self, crop: np.ndarray) -> np.ndarray:
        return self.extract_batch([crop])[0]


class DeepPatchMemoryBank:
    """PatchCore memory bank with normal-only threshold calibration."""
    def __init__(self, max_patches: int = 20000, seed: int = 42, sampling_method: str = 'random',
                 coreset_projection_dim: int = 64, coreset_candidate_patches: int = 30000,
                 coreset_batch_size: int = 2048, coreset_greedy_steps: int = 2048):
        self.max_patches = int(max_patches)
        self.seed = int(seed)
        self.sampling_method = str(sampling_method).lower()
        self.coreset_projection_dim = int(coreset_projection_dim)
        self.coreset_candidate_patches = int(coreset_candidate_patches)
        self.coreset_batch_size = int(coreset_batch_size)
        self.coreset_greedy_steps = int(coreset_greedy_steps)
        self.memory: Optional[np.ndarray] = None
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None
        self.train_roi_scores: Optional[np.ndarray] = None

    def _sample_memory_patches(self, patches: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, str, int]:
        if len(patches) <= self.max_patches:
            return patches, 'all', int(len(patches))
        if self.sampling_method in ('random', 'rand'):
            keep = rng.choice(len(patches), size=self.max_patches, replace=False)
            return patches[keep], 'random', int(len(patches))
        if self.sampling_method not in ('coreset', 'greedy_coreset', 'approx_greedy_coreset'):
            raise ValueError(f'Unsupported memory_sampling_method: {self.sampling_method}')
        keep = self._approx_greedy_coreset_indices(patches, rng)
        return patches[keep], 'approx_greedy_coreset', int(len(patches))

    def _approx_greedy_coreset_indices(self, patches: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n, d = patches.shape
        target = min(self.max_patches, n)
        candidate_n = min(n, max(target, self.coreset_candidate_patches))
        candidate_idx = rng.choice(n, size=candidate_n, replace=False) if candidate_n < n else np.arange(n)
        cand = patches[candidate_idx].astype(np.float32, copy=False)

        proj_dim = max(1, min(self.coreset_projection_dim, d))
        if proj_dim < d:
            projection = rng.normal(0.0, 1.0 / np.sqrt(proj_dim), size=(d, proj_dim)).astype(np.float32)
            work = cand @ projection
        else:
            work = cand.copy()
        work_mean = work.mean(axis=0, keepdims=True)
        work_std = work.std(axis=0, keepdims=True) + 1e-6
        work = ((work - work_mean) / work_std).astype(np.float32)

        greedy_steps = min(target, candidate_n, max(1, int(self.coreset_greedy_steps)))
        selected = np.empty(greedy_steps, dtype=np.int64)
        selected_mask = np.zeros(candidate_n, dtype=bool)
        first = int(rng.integers(candidate_n))
        selected[0] = first
        selected_mask[first] = True
        min_dist2 = self._dist2_to_point(work, work[first])
        batch = max(1, int(self.coreset_batch_size))
        for i in range(1, greedy_steps):
            farthest = int(np.argmax(min_dist2))
            selected[i] = farthest
            selected_mask[farthest] = True
            point = work[farthest]
            for start in range(0, candidate_n, batch):
                end = min(candidate_n, start + batch)
                diff = work[start:end] - point
                dist2 = np.einsum('ij,ij->i', diff, diff)
                min_dist2[start:end] = np.minimum(min_dist2[start:end], dist2)
        if greedy_steps < target:
            remaining = np.flatnonzero(~selected_mask)
            fill_n = min(target - greedy_steps, len(remaining))
            if fill_n > 0:
                far_remaining = remaining[np.argpartition(min_dist2[remaining], -fill_n)[-fill_n:]]
                selected = np.concatenate([selected, far_remaining.astype(np.int64)])
        return candidate_idx[selected[:target]]

    @staticmethod
    def _dist2_to_point(x: np.ndarray, point: np.ndarray) -> np.ndarray:
        diff = x - point
        return np.einsum('ij,ij->i', diff, diff).astype(np.float32)

    def fit(self, roi_patch_features: List[np.ndarray], threshold_quantile: float = 0.99,
            threshold_margin: float = 1.05, top_percent: float = 0.01,
            min_validation_count: int = 20) -> Dict:
        if len(roi_patch_features) < 10:
            raise RuntimeError(f'Too few normal ROI feature sets: {len(roi_patch_features)}')
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(roi_patch_features))
        roi_patch_features = [roi_patch_features[i] for i in order]
        n_val = max(min_validation_count, int(round(0.20 * len(roi_patch_features))))
        n_val = min(max(1, n_val), max(1, len(roi_patch_features) // 2))
        val_rois = roi_patch_features[:n_val]
        mem_rois = roi_patch_features[n_val:]
        if len(mem_rois) < 5:
            mem_rois = roi_patch_features
            val_rois = roi_patch_features
        patches = np.concatenate(mem_rois, axis=0).astype(np.float32)
        source_patch_count = int(len(patches))
        patches, sampling_method, candidate_patch_count = self._sample_memory_patches(patches, rng)
        self.mean = patches.mean(axis=0, keepdims=True).astype(np.float32)
        self.std = (patches.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
        self.memory = ((patches - self.mean) / self.std).astype(np.float32)
        val_scores = np.asarray([self.score_roi(p, top_percent=top_percent) for p in val_rois], np.float32)
        self.train_roi_scores = val_scores
        self.threshold = float(np.quantile(val_scores, threshold_quantile) * threshold_margin)
        return {
            'normal_roi_count': int(len(roi_patch_features)),
            'memory_patch_count': int(len(self.memory)),
            'source_patch_count': source_patch_count,
            'candidate_patch_count': candidate_patch_count,
            'memory_sampling_method': sampling_method,
            'val_roi_count': int(len(val_rois)),
            'patch_dim': int(patches.shape[1]),
            'threshold': float(self.threshold),
            'normal_score_mean': float(val_scores.mean()),
            'normal_score_q95': float(np.quantile(val_scores, 0.95)),
            'normal_score_q99': float(np.quantile(val_scores, min(0.99, threshold_quantile))),
        }

    def _score_patches_z(self, z: np.ndarray, chunk: int = 512) -> np.ndarray:
        if self.memory is None:
            raise RuntimeError('Memory bank not fitted.')
        z = np.asarray(z, np.float32)
        mem = np.asarray(self.memory, np.float32)
        mem_norm = (mem * mem).sum(axis=1)[None, :]
        out = []
        for i in range(0, len(z), chunk):
            q = z[i:i + chunk]
            q_norm = (q * q).sum(axis=1)[:, None]
            dist2 = q_norm + mem_norm - 2.0 * (q @ mem.T)
            dist2 = np.maximum(dist2, 0.0)
            out.append(np.sqrt(dist2.min(axis=1)))
        return np.concatenate(out).astype(np.float32)

    def score_roi(self, patch_features: np.ndarray, top_percent: float = 0.01) -> float:
        if self.memory is None or self.mean is None or self.std is None:
            raise RuntimeError('Memory bank not fitted.')
        z = ((patch_features.astype(np.float32) - self.mean) / self.std).astype(np.float32)
        patch_scores = self._score_patches_z(z)
        k = max(1, int(round(len(patch_scores) * top_percent)))
        top = np.partition(patch_scores, -k)[-k:]
        return float(top.mean())

    def score_roi_with_map(self, patch_features: np.ndarray, top_percent: float = 0.01) -> tuple[float, np.ndarray]:
        if self.memory is None or self.mean is None or self.std is None:
            raise RuntimeError('Memory bank not fitted.')
        z = ((patch_features.astype(np.float32) - self.mean) / self.std).astype(np.float32)
        patch_scores = self._score_patches_z(z)
        k = max(1, int(round(len(patch_scores) * top_percent)))
        top = np.partition(patch_scores, -k)[-k:]
        return float(top.mean()), patch_scores.astype(np.float32)

    def to_npz_dict(self, prefix: str) -> Dict[str, np.ndarray]:
        if self.memory is None or self.mean is None or self.std is None or self.threshold is None:
            raise RuntimeError('Memory bank not fitted.')
        return {
            f'{prefix}_memory': self.memory,
            f'{prefix}_mean': self.mean,
            f'{prefix}_std': self.std,
            f'{prefix}_threshold': np.asarray([self.threshold], np.float32),
            f'{prefix}_train_roi_scores': np.asarray(self.train_roi_scores if self.train_roi_scores is not None else [], np.float32),
        }

    @classmethod
    def from_npz_dict(cls, data, prefix: str, max_patches: int = 20000, seed: int = 42,
                      sampling_method: str = 'random', coreset_projection_dim: int = 128,
                      coreset_candidate_patches: int = 60000,
                      coreset_batch_size: int = 1024,
                      coreset_greedy_steps: int = 2048) -> 'DeepPatchMemoryBank':
        bank = cls(max_patches=max_patches, seed=seed, sampling_method=sampling_method,
                   coreset_projection_dim=coreset_projection_dim,
                   coreset_candidate_patches=coreset_candidate_patches,
                   coreset_batch_size=coreset_batch_size,
                   coreset_greedy_steps=coreset_greedy_steps)
        bank.memory = data[f'{prefix}_memory'].astype(np.float32)
        bank.mean = data[f'{prefix}_mean'].astype(np.float32)
        bank.std = data[f'{prefix}_std'].astype(np.float32)
        bank.threshold = float(data[f'{prefix}_threshold'][0])
        k = f'{prefix}_train_roi_scores'
        if k in data:
            bank.train_roi_scores = data[k].astype(np.float32)
        return bank
