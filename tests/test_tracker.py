"""Геометрия на известном преобразовании, потеря и повторное обнаружение."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracker import PlanarTracker
from evaluate import polygon_iou


class TrackerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(4)
        rng = np.random.default_rng(42)
        cls.frame = np.zeros((360, 480, 3), np.uint8)
        texture = rng.integers(0, 256, (220, 300, 3), np.uint8)
        texture = cv2.GaussianBlur(texture, (3, 3), 0)
        cls.frame[60:280, 80:380] = texture
        cv2.putText(cls.frame, 'KEYPOINTS', (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cls.quad = np.float32([[80,60],[379,60],[379,279],[80,279]])

    def test_known_perspective(self):
        tracker = PlanarTracker(self.frame, quad=self.quad)
        target = np.float32([[100,50],[395,80],[365,300],[60,260]])
        h = cv2.getPerspectiveTransform(self.quad, target)
        transformed = cv2.warpPerspective(self.frame, h, (480,360))
        result = tracker.update(transformed)
        self.assertEqual(result.mode, 'detect')
        self.assertGreater(polygon_iou(result.polygon, target), .97)

    def test_loss_has_no_old_polygon_and_reacquires(self):
        tracker = PlanarTracker(self.frame, quad=self.quad)
        lost = tracker.update(np.zeros_like(self.frame))
        self.assertEqual(lost.mode, 'lost')
        self.assertIsNone(lost.polygon)
        found = tracker.update(self.frame)
        self.assertEqual(found.mode, 'detect')
        self.assertGreater(polygon_iou(found.polygon, self.quad), .99)

    def test_invalid_initialization(self):
        with self.assertRaises(ValueError):
            PlanarTracker(self.frame, roi=(400,0,200,100))
        with self.assertRaises(ValueError):
            PlanarTracker(self.frame, quad=[[0,0],[100,100],[100,0],[0,100]])
        with self.assertRaises(ValueError):
            PlanarTracker(np.zeros_like(self.frame))


if __name__ == '__main__':
    unittest.main()
