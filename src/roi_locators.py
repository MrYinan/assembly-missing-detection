# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class VideoBRoiConfig:
    base_width: int = 1280
    base_height: int = 720
    blue_h_min: int = 90
    blue_h_max: int = 135
    top_center_y_min: int = 210
    top_center_y_max: int = 285
    min_blue_area: int = 650
    min_product_center_x: int = 520
    stable_center_x_min: int = 720
    stable_center_x_max: int = 1245
    max_products_per_frame: int = 1
    inspection_center_x: int = 970
    inspection_selection: str = 'rightmost'
    port_x_offset: int = -74
    port_w: int = 145
    port_h: int = 82
    top_port_center_y: int = 238
    bottom_port_center_y: int = 638
    min_roi_x: int = 640
    cap_roi_pad_x: int = 24
    cap_roi_pad_y: int = 15
    cap_roi_pad_left: int = 56
    cap_roi_pad_right: int = 30
    cap_roi_pad_top: int = 22
    cap_roi_pad_bottom: int = 22
    top_roi_extra_down: int = 28
    bottom_roi_extra_up: int = 28
    cap_center_match_tolerance: int = 90


class VideoBSidePortRoiLocator:
    """OpenCV is used only for ROI localisation, not as the anomaly model."""
    def __init__(self, cfg: VideoBRoiConfig | None = None):
        self.cfg = cfg or VideoBRoiConfig()

    def _scale(self, frame: np.ndarray) -> Tuple[float, float]:
        h, w = frame.shape[:2]
        return w / max(float(self.cfg.base_width), 1.0), h / max(float(self.cfg.base_height), 1.0)

    def _blue_components(self, frame: np.ndarray) -> List[Dict]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([self.cfg.blue_h_min, 45, 30]), np.array([self.cfg.blue_h_max, 255, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, _lab, stats, cent = cv2.connectedComponentsWithStats(mask)
        comps: List[Dict] = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            cx, cy = cent[i]
            comps.append({
                'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
                'area': int(area), 'cx': float(cx), 'cy': float(cy),
            })
        return comps

    def _is_cap_component(self, comp: Dict, sx: float, sy: float) -> bool:
        min_area = max(60, int(round(self.cfg.min_blue_area * sx * sy * 0.45)))
        w, h, area = int(comp['w']), int(comp['h']), int(comp['area'])
        return (
            area >= min_area and
            max(18, int(round(35 * sx))) <= w <= max(45, int(round(170 * sx))) and
            max(10, int(round(25 * sy))) <= h <= max(24, int(round(95 * sy)))
        )

    def detect_product_centers(self, frame: np.ndarray) -> List[int]:
        sx, sy = self._scale(frame)
        xs: List[int] = []
        H, W = frame.shape[:2]
        y_min = self.cfg.top_center_y_min * sy
        y_max = self.cfg.top_center_y_max * sy
        min_center_x = self.cfg.min_product_center_x * sx
        for comp in self._blue_components(frame):
            cx, cy = comp['cx'], comp['cy']
            if not self._is_cap_component(comp, sx, sy):
                continue
            if not (y_min <= cy <= y_max):
                continue
            if not (min_center_x <= cx <= W - 10):
                continue
            xs.append(int(round(cx)))
        xs = sorted(xs)
        merged: List[int] = []
        for x in xs:
            if not merged or x - merged[-1] > max(25, 50 * sx):
                merged.append(x)
            else:
                merged[-1] = int(round((merged[-1] + x) / 2))
        return merged

    def _component_bbox(self, frame: np.ndarray, comp: Dict, position: str = '') -> List[int]:
        H, W = frame.shape[:2]
        sx, sy = self._scale(frame)
        pad_left = int(round(getattr(self.cfg, 'cap_roi_pad_left', self.cfg.cap_roi_pad_x) * sx))
        pad_right = int(round(getattr(self.cfg, 'cap_roi_pad_right', self.cfg.cap_roi_pad_x) * sx))
        pad_top = int(round(getattr(self.cfg, 'cap_roi_pad_top', self.cfg.cap_roi_pad_y) * sy))
        pad_bottom = int(round(getattr(self.cfg, 'cap_roi_pad_bottom', self.cfg.cap_roi_pad_y) * sy))
        if position == 'top':
            pad_bottom += int(round(self.cfg.top_roi_extra_down * sy))
        elif position == 'bottom':
            pad_top += int(round(self.cfg.bottom_roi_extra_up * sy))
        x0 = max(0, int(comp['x']) - pad_left)
        y0 = max(0, int(comp['y']) - pad_top)
        x1 = min(W, int(comp['x']) + int(comp['w']) + pad_right)
        y1 = min(H, int(comp['y']) + int(comp['h']) + pad_bottom)
        return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]

    def _fixed_bbox(self, frame: np.ndarray, cx: int, cy0: int, position: str = '') -> List[int]:
        H, W = frame.shape[:2]
        sx, sy = self._scale(frame)
        pad_left_extra = max(0, self.cfg.cap_roi_pad_left - self.cfg.cap_roi_pad_x)
        pad_right_extra = max(0, self.cfg.cap_roi_pad_right - self.cfg.cap_roi_pad_x)
        pad_top_extra = max(0, self.cfg.cap_roi_pad_top - self.cfg.cap_roi_pad_y)
        pad_bottom_extra = max(0, self.cfg.cap_roi_pad_bottom - self.cfg.cap_roi_pad_y)
        if position == 'top':
            pad_bottom_extra += self.cfg.top_roi_extra_down
        elif position == 'bottom':
            pad_top_extra += self.cfg.bottom_roi_extra_up
        base_w = max(18, int(round(self.cfg.port_w * sx)))
        base_h = max(18, int(round(self.cfg.port_h * sy)))
        x0 = int(round(cx + self.cfg.port_x_offset * sx))
        y0 = int(round(cy0 * sy - base_h / 2))
        x = int(round(x0 - pad_left_extra * sx))
        y = int(round(y0 - pad_top_extra * sy))
        port_w = int(round(base_w + (pad_left_extra + pad_right_extra) * sx))
        port_h = int(round(base_h + (pad_top_extra + pad_bottom_extra) * sy))
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        return [x, y, min(port_w, W - x), min(port_h, H - y)]

    def _nearest_cap_component(self, comps: List[Dict], cx: int, y_min: float, y_max: float,
                               sx: float, sy: float) -> Dict | None:
        tol = max(40.0, self.cfg.cap_center_match_tolerance * sx)
        candidates = [
            c for c in comps
            if self._is_cap_component(c, sx, sy)
            and y_min <= float(c['cy']) <= y_max
            and abs(float(c['cx']) - float(cx)) <= tol
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda c: abs(float(c['cx']) - float(cx)))

    def _select_inspection_centers(self, centers: List[int], sx: float) -> List[int]:
        max_products = int(getattr(self.cfg, 'max_products_per_frame', 0))
        if max_products <= 0 or len(centers) <= max_products:
            return centers
        mode = str(getattr(self.cfg, 'inspection_selection', 'nearest')).lower()
        if mode in ('rightmost', 'right', 'max_x'):
            ranked = sorted(centers, reverse=True)
        elif mode in ('leftmost', 'left', 'min_x'):
            ranked = sorted(centers)
        else:
            target_x = float(self.cfg.inspection_center_x) * sx
            ranked = sorted(centers, key=lambda x: (abs(float(x) - target_x), float(x)))
        return sorted(ranked[:max_products])

    def detect_rois(self, frame: np.ndarray, stable_only: bool = True) -> List[Dict]:
        H, W = frame.shape[:2]
        sx, sy = self._scale(frame)
        stable_min = self.cfg.stable_center_x_min * sx
        stable_max = self.cfg.stable_center_x_max * sx
        min_roi_x = self.cfg.min_roi_x * sx
        comps = self._blue_components(frame)
        top_y_min = self.cfg.top_center_y_min * sy
        top_y_max = self.cfg.top_center_y_max * sy
        bottom_y_min = (self.cfg.bottom_port_center_y - self.cfg.port_h) * sy
        bottom_y_max = (self.cfg.bottom_port_center_y + self.cfg.port_h) * sy
        rois: List[Dict] = []
        centers = self.detect_product_centers(frame)
        if stable_only:
            centers = [cx for cx in centers if stable_min <= cx <= stable_max]
        centers = self._select_inspection_centers(centers, sx)
        for cx in centers:
            if stable_only and not (stable_min <= cx <= stable_max):
                continue
            for pos, cy0, y_min, y_max in (
                    ('top', self.cfg.top_port_center_y, top_y_min, top_y_max),
                    ('bottom', self.cfg.bottom_port_center_y, bottom_y_min, bottom_y_max)):
                comp = self._nearest_cap_component(comps, cx, y_min, y_max, sx, sy)
                x, y, w, h = self._component_bbox(frame, comp, pos) if comp else self._fixed_bbox(frame, cx, cy0, pos)
                if x < min_roi_x or y < 0 or x + w >= W or y + h >= H:
                    continue
                rois.append({'position': pos, 'center_x': int(cx), 'bbox': [x, y, int(w), int(h)]})
        return rois


@dataclass
class VideoARoiConfig:
    # Coordinates below are defined on a 1280x720 reference frame and are
    # automatically scaled to the actual video resolution.  This is important
    # because the original Video A is 1920x1080; without scaling, the old
    # locator used a y/radius range that was too small and many normal caps
    # were not drawn in the rendered video.
    base_width: int = 1280
    base_height: int = 720
    x_min: int = 410
    x_max: int = 1270
    y_min: int = 230
    y_max: int = 455
    min_radius: int = 36
    max_radius: int = 92
    max_faces: int = 3
    inspection_center_x: int = 600
    inspection_selection: str = 'strongest'
    yellow_h_min: int = 14
    yellow_h_max: int = 45
    # Video A circle candidates are first generated by Hough transform.  The
    # following checks remove transparent handles, metal screw holes and
    # background circular patterns.  They do not use defect labels; they only
    # describe what a visible product front face looks like.
    min_inner_saturation: float = 45.0
    min_face_yellow_ratio: float = 0.10
    nms_radius_factor: float = 1.35


class VideoACircleRoiLocator:
    def __init__(self, cfg: VideoARoiConfig | None = None):
        self.cfg = cfg or VideoARoiConfig()

    def _scale(self, frame: np.ndarray) -> Tuple[float, float]:
        h, w = frame.shape[:2]
        return w / max(float(self.cfg.base_width), 1.0), h / max(float(self.cfg.base_height), 1.0)

    def yellow_ratio(self, crop: np.ndarray) -> float:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        return float(((H >= self.cfg.yellow_h_min) & (H <= self.cfg.yellow_h_max) & (S > 70) & (V > 70)).mean())

    def _face_candidate_metrics(self, crop: np.ndarray, local_cx: int, local_cy: int, r: int) -> Dict[str, float]:
        """Return simple geometry/color metrics for a Hough circle candidate.

        This is still only ROI localisation.  The OK/NG decision is made later
        by Deep PatchCore.  The metrics are used to reject non-product circles
        such as transparent handles or background holes, which caused boxes to
        drift away from the real cap face in Video A.
        """
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        yy, xx = np.ogrid[:crop.shape[0], :crop.shape[1]]
        rr = np.sqrt((xx - local_cx) ** 2 + (yy - local_cy) ** 2)
        inner = rr < max(3, r * 0.45)
        inner_s = float(sat[inner].mean()) if inner.any() else 0.0
        yr = self.yellow_ratio(crop)
        # A real front face can be normal yellow or pale/peach defective, but
        # it should not look like a nearly gray transparent handle.
        face_score = yr + inner_s / 255.0
        return {'yellow_ratio': yr, 'inner_saturation': inner_s, 'face_score': float(face_score)}

    def _select_inspection_candidates(self, candidates: List[Dict], sx: float) -> List[Dict]:
        max_faces = int(getattr(self.cfg, 'max_faces', 0))
        if max_faces <= 0 or len(candidates) <= max_faces:
            return candidates
        mode = str(getattr(self.cfg, 'inspection_selection', 'strongest')).lower()
        if mode in ('nearest', 'fixed', 'station'):
            target_x = float(getattr(self.cfg, 'inspection_center_x', 600)) * sx
            ranked = sorted(candidates, key=lambda d: (abs(float(d['circle'][0]) - target_x), -float(d.get('face_score', 0.0))))
        elif mode in ('rightmost', 'right', 'max_x'):
            ranked = sorted(candidates, key=lambda d: float(d['circle'][0]), reverse=True)
        elif mode in ('leftmost', 'left', 'min_x'):
            ranked = sorted(candidates, key=lambda d: float(d['circle'][0]))
        else:
            # Keep the original behavior: strongest face-like candidates first.
            ranked = sorted(candidates, key=lambda d: (-float(d.get('face_score', 0.0)), float(d['circle'][0])))
        return sorted(ranked[:max_faces], key=lambda d: d['circle'][0])

    def detect_rois(self, frame: np.ndarray, stable_only: bool = True) -> List[Dict]:
        sx, sy = self._scale(frame)
        # Scale ROI search region/radius from the 1280x720 reference to the
        # actual video size.  Video A is often 1920x1080, so this directly fixes
        # missing boxes on normal frames.
        x_min = int(round(self.cfg.x_min * sx)); x_max = int(round(self.cfg.x_max * sx))
        y_min = int(round(self.cfg.y_min * sy)); y_max = int(round(self.cfg.y_max * sy))
        # circle radius is isotropic in pixels; use average scale for robustness
        sr = (sx + sy) / 2.0
        min_r = max(8, int(round(self.cfg.min_radius * sr)))
        max_r = max(min_r + 2, int(round(self.cfg.max_radius * sr)))
        fixed_r = int(round(48 * sr))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv2.INTER_AREA)
        small = cv2.medianBlur(small, 5)
        circles = cv2.HoughCircles(
            small, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=max(30, int(round(45 * sr / 2.0))), param1=80, param2=25,
            minRadius=max(4, min_r // 2), maxRadius=max(6, max_r // 2),
        )
        H, W = frame.shape[:2]
        cands = []
        if circles is not None:
            for xs, ys, rs in np.round(circles[0]).astype(int):
                x, y, r_orig = int(xs * 2), int(ys * 2), int(rs * 2)
                if not (x_min <= x <= x_max and y_min <= y <= y_max and min_r <= r_orig <= max_r):
                    continue
                r = fixed_r
                x0, x1 = max(0, x - r), min(W, x + r)
                y0, y1 = max(0, y - r), min(H, y + r)
                crop = frame[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                m = self._face_candidate_metrics(crop, x - x0, y - y0, r)
                # Reject background/handle candidates.  This is a product-face
                # visibility gate, not an anomaly rule: it is applied to training
                # and testing frames in exactly the same way.  The threshold is
                # intentionally not too strict, so pale/defective faces are still
                # kept and can be judged by Deep PatchCore later.
                if (m['yellow_ratio'] < self.cfg.min_face_yellow_ratio and
                        m['inner_saturation'] < self.cfg.min_inner_saturation):
                    continue
                cands.append({'circle': [x, y, r], **m, 'source': 'hough'})

        # HSV color fallback for normal yellow caps.  Hough may occasionally miss
        # the front face when there is glare or motion blur.  This fallback is
        # used only to localise a candidate ROI; the OK/NG decision still comes
        # from Deep PatchCore / yellow gate after training.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([self.cfg.yellow_h_min, 35, 60]), np.array([self.cfg.yellow_h_max, 255, 255]))
        region = np.zeros(mask.shape, np.uint8)
        region[max(0, y_min):min(H, y_max), max(0, x_min):min(W, x_max)] = 255
        mask = cv2.bitwise_and(mask, region)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = np.pi * (min_r * 0.35) ** 2
        max_area = np.pi * (max_r * 0.95) ** 2
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            (x, y), r0 = cv2.minEnclosingCircle(cnt)
            x, y = int(round(x)), int(round(y))
            # The visible yellow disk is inside the transparent cap, so expand
            # radius to approximate the whole front-face ROI.
            r = fixed_r
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                continue
            x0, x1 = max(0, x - r), min(W, x + r)
            y0, y1 = max(0, y - r), min(H, y + r)
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            m = self._face_candidate_metrics(crop, x - x0, y - y0, r)
            cands.append({'circle': [x, y, r], **m, 'source': 'yellow'})

        # Prefer true front faces.  Stronger NMS removes secondary Hough circles
        # on transparent side handles near a real product face.  Keep left-to-right
        # order for visualization and product rows.
        cands = sorted(cands, key=lambda d: (-d['face_score'], d['circle'][0]))
        selected: List[Dict] = []
        for c in cands:
            x, y, r = c['circle']
            keep = True
            for s in selected:
                sx0, sy0, sr0 = s['circle']
                if (x - sx0) ** 2 + (y - sy0) ** 2 <= (self.cfg.nms_radius_factor * max(r, sr0)) ** 2:
                    keep = False
                    break
            if keep:
                selected.append(c)
        selected = self._select_inspection_candidates(selected, sx)
        return sorted(selected, key=lambda d: d['circle'][0])
