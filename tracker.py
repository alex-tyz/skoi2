"""Трекинг плоского объекта: SIFT, RANSAC и резервный поток Lucas–Kanade.

Исходная идея и прототип: skadibtw/CV_course, lab2 (источники в README.md).
Здесь добавлены четырёхугольный эталон, банк масштабов, проверка
пространственного покрытия, диагностика и отказ от устаревшей рамки.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TrackingResult:
    polygon: np.ndarray | None
    mode: str
    matches: int = 0
    inliers: int = 0
    reprojection_error: float | None = None


class PlanarTracker:
    def __init__(self, frame, roi=None, quad=None):
        if roi is not None and quad is not None:
            raise ValueError('Укажите либо roi, либо quad')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.height, self.width = gray.shape
        if quad is None:
            x, y, w, h = roi if roi is not None else (0, 0, self.width, self.height)
            if x < 0 or y < 0 or w < 2 or h < 2 or x+w > self.width or y+h > self.height:
                raise ValueError('ROI должна находиться внутри первого кадра')
            quad = [[x, y], [x+w-1, y], [x+w-1, y+h-1], [x, y+h-1]]
        self.corners = np.asarray(quad, np.float32).reshape(4, 1, 2)
        if (not np.isfinite(self.corners).all()
                or not cv2.isContourConvex(self.corners)
                or cv2.contourArea(self.corners) < 100):
            raise ValueError('Нужен выпуклый четырёхугольник: углы по периметру')
        if self.valid_polygon(np.eye(3)) is None:
            raise ValueError('Начальная рамка недопустима')
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, np.int32(self.corners), 255)
        mask = cv2.erode(mask, np.ones((7, 7), np.uint8))
        self.sift = cv2.SIFT_create(nfeatures=2000)
        # Уменьшенные эталоны улучшают повторное обнаружение далёкого объекта.
        # Все точки возвращаются в систему координат первого кадра.
        reference, descriptors = [], []
        for scale in (1.0, 0.5, 0.25):
            size = (max(1, round(self.width*scale)), max(1, round(self.height*scale)))
            image = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
            scaled_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
            kp, desc = self.sift.detectAndCompute(image, scaled_mask)
            if desc is not None:
                points = np.float32([k.pt for k in kp])
                points[:, 0] *= self.width / size[0]
                points[:, 1] *= self.height / size[1]
                reference.append(points)
                descriptors.append(desc)
        if not descriptors or sum(len(d) for d in descriptors) < 12:
            raise ValueError('Слишком мало ключевых точек: выберите текстурный объект')
        self.reference_points = np.vstack(reference)
        self.descriptors = np.vstack(descriptors)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.matrix = np.eye(3)
        self.previous = gray
        self.misses = 0
        self.points = self.anchors = None
        self.seed(gray)

    def valid_polygon(self, matrix):
        if matrix is None or not np.isfinite(matrix).all():
            return None
        polygon = cv2.perspectiveTransform(self.corners, matrix)
        if not np.isfinite(polygon).all() or not cv2.isContourConvex(polygon):
            return None
        if not 100 < cv2.contourArea(polygon) < 1.5*self.width*self.height:
            return None
        vertices = polygon.reshape(4, 2)
        sides = np.linalg.norm(vertices - np.roll(vertices, 1, axis=0), axis=1)
        if sides.min() < 3 or sides.max()/sides.min() > 10:
            return None
        if (vertices[:, 0].min() < -self.width*.25
                or vertices[:, 0].max() > self.width*1.25
                or vertices[:, 1].min() < -self.height*.25
                or vertices[:, 1].max() > self.height*1.25):
            return None
        return polygon

    def estimate(self, source, target, minimum):
        if len(source) < minimum:
            return None
        matrix, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
        if matrix is None or mask is None or self.valid_polygon(matrix) is None:
            return None
        keep = mask.ravel().astype(bool)
        if keep.sum() < minimum or keep.mean() < .35:
            return None
        src = source.reshape(-1, 2)[keep]
        # Совпадения должны покрывать часть объекта, а не одну букву/угол.
        coverage = cv2.contourArea(cv2.convexHull(src)) / cv2.contourArea(self.corners)
        if coverage < .01:
            return None
        projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        error = np.linalg.norm(projected - target.reshape(-1, 2)[keep], axis=1)
        return matrix, keep, float(np.median(error))

    def detect(self, gray):
        kp, desc = self.sift.detectAndCompute(gray, None)
        if desc is None or len(kp) < 12:
            return None, 0
        pairs = self.matcher.knnMatch(self.descriptors, desc, k=2)
        candidates = [p[0] for p in pairs if len(p) == 2 and p[0].distance < .75*p[1].distance]
        # Один текущий признак не должен голосовать несколько раз из-за банка масштабов.
        unique = {}
        for match in sorted(candidates, key=lambda m: m.distance):
            unique.setdefault(match.trainIdx, match)
        good = list(unique.values())
        if len(good) < 12:
            return None, len(good)
        source = self.reference_points[[m.queryIdx for m in good]]
        target = np.float32([kp[m.trainIdx].pt for m in good])
        return self.estimate(source, target, 12), len(good)

    def seed(self, gray):
        self.points = self.anchors = None
        polygon = self.valid_polygon(self.matrix)
        if polygon is None:
            return
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, np.int32(polygon), 255)
        mask = cv2.erode(mask, np.ones((7, 7), np.uint8))
        points = cv2.goodFeaturesToTrack(gray, 250, .005, 5, mask=mask)
        if points is None:
            return
        try:
            anchors = cv2.perspectiveTransform(points, np.linalg.inv(self.matrix))
        except np.linalg.LinAlgError:
            return
        if np.isfinite(anchors).all():
            self.points, self.anchors = points, anchors

    def follow(self, gray):
        if self.points is None or len(self.points) < 8:
            return None
        options = dict(winSize=(21, 21), maxLevel=4,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        moved, status, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, self.points, None, **options)
        if moved is None or status is None:
            return None
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, moved, None, **options)
        if back is None or back_status is None:
            return None
        error = np.linalg.norm(back-self.points, axis=2).ravel()
        valid = (status.ravel() == 1) & (back_status.ravel() == 1) & (error < 1.5)
        valid &= np.isfinite(moved).all(axis=(1, 2))
        estimate = self.estimate(self.anchors[valid], moved[valid], 8)
        if estimate is None:
            return None
        matrix, keep, residual = estimate
        self.points, self.anchors = moved[valid][keep], self.anchors[valid][keep]
        return matrix, keep, residual

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        estimate, matches = self.detect(gray)
        mode = 'detect'
        if estimate is None and self.matrix is not None:
            estimate = self.follow(gray)
            mode = 'flow'
        if estimate is None:
            # Рамку скрываем. Краткий смаз переживаем предварительным
            # пересевом внутри последней области, не выдавая её за измерение.
            self.misses += 1
            if self.misses > 5:
                self.matrix = self.points = self.anchors = None
            elif self.matrix is not None:
                self.seed(gray)
                self.previous = gray
            result = TrackingResult(None, 'lost', matches)
        else:
            self.misses = 0
            self.matrix, keep, residual = estimate
            if mode == 'detect' or self.points is None or len(self.points) < 80:
                self.seed(gray)
            result = TrackingResult(self.valid_polygon(self.matrix), mode,
                                    matches, int(keep.sum()), residual)
            self.previous = gray
        return result
