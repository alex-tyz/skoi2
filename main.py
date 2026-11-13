"""Чтение видео, выбор эталона, аннотация, покадровые метрики и MP4."""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from tracker import PlanarTracker, TrackingResult


def write_image(path, image):
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f'Не удалось сохранить {path}')
    encoded.tofile(path)


def annotate(frame, result, label, index):
    output = frame.copy()
    color = (50, 220, 50) if result.mode != 'flow' else (0, 220, 255)
    if result.polygon is not None:
        cv2.polylines(output, [np.int32(result.polygon)], True, color, 3, cv2.LINE_AA)
        x, y = result.polygon.reshape(4, 2).min(axis=0)
        position = (int(np.clip(x, 0, frame.shape[1]-80)), int(np.clip(y-10, 28, frame.shape[0]-10)))
        cv2.putText(output, label, position, cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(output, label, position, cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2, cv2.LINE_AA)
    info = f'{index} | {result.mode} | inliers={result.inliers}'
    pos = (10, frame.shape[0]-18)
    cv2.putText(output, info, pos, cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(output, info, pos, cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def process_video(video, output_dir, label='Object', roi=None, quad=None, show=False,
                  select=False, transcode=True):
    video, output_dir = Path(video), Path(output_dir)
    if not video.is_file():
        raise ValueError(f'Файл не найден: {video}')
    if not label.isascii():
        raise ValueError('Подпись должна быть латиницей (шрифт OpenCV)')
    capture = cv2.VideoCapture(str(video))
    writer = None
    try:
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f'Не удалось прочитать видео: {video}')
        if select:
            roi = tuple(map(int, cv2.selectROI('Select object, Enter to confirm', frame, False, False)))
            cv2.destroyAllWindows()
        cv2.setRNGSeed(0)
        cv2.setNumThreads(4)
        tracker = PlanarTracker(frame, roi=roi, quad=quad)
        height, width = frame.shape[:2]
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_ids = set(np.linspace(0, max(0, expected-1), 8).astype(int))
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f'{video.stem}_tracked.mp4'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not writer.isOpened():
            raise ValueError(f'Не удалось создать видео: {path}')
        modes = dict(initial=0, detect=0, flow=0, lost=0)
        rows, samples = [], []
        source_quad = tracker.corners.reshape(4, 2).tolist()
        write_image(output_dir / f'{video.stem}_initial.jpg',
                    annotate(frame, TrackingResult(tracker.corners, 'initial'), label, 0))
        started = perf_counter()
        complete = True
        while ok:
            idx = len(rows)
            result = (TrackingResult(tracker.corners.copy(), 'initial') if idx == 0
                      else tracker.update(frame))
            modes[result.mode] += 1
            polygon = result.polygon.reshape(4, 2).ravel().tolist() if result.polygon is not None else [None]*8
            rows.append([idx, idx/fps, result.mode, result.matches, result.inliers,
                         result.reprojection_error, *polygon])
            decorated = annotate(frame, result, label, idx)
            writer.write(decorated)
            if idx in sample_ids:
                scale = min(480/width, 360/height)
                thumbnail = cv2.resize(decorated, None, fx=scale, fy=scale)
                samples.append(thumbnail)
            if show:
                cv2.imshow('Tracking - Esc to stop', decorated)
                if cv2.waitKey(1) & 0xFF == 27:
                    complete = False
                    break
            ok, frame = capture.read()
        elapsed = perf_counter()-started
        writer.release()
        writer = None
        # H.264 обеспечивает просмотр MP4 в браузере и GitHub.
        # Исходное видео не изменяется; обработанный ролик сохраняется без звука.
        codec = 'mp4v'
        ffmpeg = shutil.which('ffmpeg') if transcode else None
        encoding_time = 0.0
        if ffmpeg:
            converted = output_dir / f'{video.stem}_h264.tmp.mp4'
            start = perf_counter()
            proc = subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                                   '-i', str(path), '-an', '-c:v', 'libx264', '-preset', 'fast',
                                   '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                                   str(converted)], capture_output=True, text=True)
            encoding_time = perf_counter()-start
            if proc.returncode:
                raise RuntimeError(f'Ошибка FFmpeg: {proc.stderr}')
            converted.replace(path)
            codec = 'h264'
        if samples:
            while len(samples) < 8:
                samples.append(samples[-1])
            preview = np.vstack([np.hstack(samples[:4]), np.hstack(samples[4:8])])
            write_image(output_dir / f'{video.stem}_preview.jpg', preview)
        fields = ['frame', 'time_seconds', 'mode', 'matches', 'inliers', 'median_reprojection_px']
        fields += [f'p{i}{axis}' for i in range(4) for axis in ('x', 'y')]
        with (output_dir / f'{video.stem}_frames.csv').open('w', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(fields)
            csv_writer.writerows(rows)
        updated = modes['detect']+modes['flow']
        errors = [r[5] for r in rows if r[5] is not None]
        stats = dict(video=video.name, width=width, height=height, input_fps=fps,
                     video_duration_seconds=round(len(rows)/fps, 3), expected_frames=expected,
                     frames=len(rows), complete=complete and len(rows) == expected,
                     initial_quad=source_quad, label=label, reference_descriptors=len(tracker.descriptors),
                     modes=modes, frames_with_polygon=updated+modes['initial'],
                     updated_frames=updated,
                     updated_after_initial_percent=round(100*updated/max(1, len(rows)-1), 2),
                     processing_seconds=round(elapsed, 3), processing_fps=round(len(rows)/elapsed, 2),
                     encoding_seconds=round(encoding_time, 3), output_codec=codec,
                     median_reprojection_px=round(float(np.median(errors)), 4) if errors else None,
                     versions=dict(python=sys.version.split()[0], opencv=cv2.__version__, numpy=np.__version__),
                     parameters=dict(reference_scales=[1, .5, .25], sift_features=2000, ratio=.75,
                                     ransac_px=3, detect_inliers=12, flow_inliers=8,
                                     min_inlier_ratio=.35, min_reference_coverage=.01,
                                     forward_backward_px=1.5, threads=4))
        (output_dir / f'{video.stem}_stats.json').write_text(
            json.dumps(stats, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        return stats
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show or select:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Трекер плоского текстурного объекта')
    parser.add_argument('video', type=Path)
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--label', default='Object')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--roi', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'))
    group.add_argument('--quad', type=float, nargs=8, help='x0 y0 x1 y1 x2 y2 x3 y3 по периметру')
    group.add_argument('--select', action='store_true', help='Выделить объект мышью на первом кадре')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--no-transcode', action='store_true', help='Оставить MP4V вместо H.264')
    args = parser.parse_args()
    try:
        stats = process_video(args.video, args.output, args.label, args.roi, args.quad,
                              args.show, args.select, not args.no_transcode)
    except (ValueError, RuntimeError, cv2.error) as exc:
        parser.exit(1, f'{exc}\n')
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
