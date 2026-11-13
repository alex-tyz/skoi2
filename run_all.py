"""Воспроизводимый запуск трёх тестов и собственного видео из любого каталога."""
import json
from pathlib import Path
from main import process_video

ROOT = Path(__file__).resolve().parent
CASES = [
    ('mona-lisa.avi', 'Mona Lisa', None),
    ('mona-lisa-blur.avi', 'Mona Lisa', None),
    ('mona-lisa-blur-extra-credit.avi', 'Mona Lisa', None),
    ('my_video.mp4', 'Book', [60, 131, 687, 130, 734, 1049, 34, 1060]),
]


def main():
    results = []
    for name, label, quad in CASES:
        result = process_video(ROOT/'data'/name, ROOT/'results', label=label, quad=quad)
        results.append(result)
        print(f"{name}: {result['modes']}, {result['processing_fps']} FPS", flush=True)
    (ROOT/'results/summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
