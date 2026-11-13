"""Выборочная оценка IoU и ошибки углов по независимой разметке."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from main import write_image

ROOT = Path(__file__).resolve().parent


def polygon_iou(a, b):
    a, b = np.asarray(a, np.float32).reshape(4, 2), np.asarray(b, np.float32).reshape(4, 2)
    intersection, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a)+cv2.contourArea(b)-intersection
    return float(intersection/union) if union > 0 else 0.0


def main():
    annotations = json.loads((ROOT/'data/book_annotations.json').read_text(encoding='utf-8'))
    with (ROOT/'results/my_video_frames.csv').open(encoding='utf-8') as f:
        predictions = {int(row['frame']): row for row in csv.DictReader(f)}
    capture = cv2.VideoCapture(str(ROOT/'data/my_video.mp4'))
    rows, tiles = [], []
    for item in annotations['frames']:
        idx, truth = item['frame'], np.float32(item['quad'])
        row = predictions[idx]
        if row['mode'] == 'lost':
            predicted, iou, error = None, 0.0, None
        else:
            predicted = np.float32([[float(row[f'p{i}x']), float(row[f'p{i}y'])] for i in range(4)])
            iou = polygon_iou(predicted, truth)
            error = float(np.linalg.norm(predicted-truth, axis=1).mean())
        rows.append(dict(frame=idx, iou=round(iou, 4), mean_corner_error_px=round(error, 2) if error is not None else None))
        capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f'Не прочитан кадр {idx}')
        cv2.polylines(frame, [np.int32(truth)], True, (255, 255, 0), 3)
        if predicted is not None:
            cv2.polylines(frame, [np.int32(predicted)], True, (40, 230, 40), 3)
        cv2.putText(frame, f'{idx} IoU={iou:.3f}', (12, 1240), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 5)
        cv2.putText(frame, f'{idx} IoU={iou:.3f}', (12, 1240), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
        tiles.append(cv2.resize(frame, (270, 480)))
    capture.release()
    report = dict(annotation_note=annotations['description'], samples=len(rows),
                  mean_iou=round(float(np.mean([r['iou'] for r in rows])), 4),
                  mean_corner_error_px=round(float(np.mean([r['mean_corner_error_px'] for r in rows if r['mean_corner_error_px'] is not None])), 2),
                  per_frame=rows)
    (ROOT/'results/book_evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    write_image(ROOT/'results/book_evaluation.jpg', np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
