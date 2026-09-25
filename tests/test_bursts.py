from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

from osprey.cli import COLUMNS, Frame, _bursts, _pick_best
from osprey.detect import Bird
from osprey.exif import PhotoMeta

T0 = datetime(2026, 8, 22, 8, 20, 38)  # noqa: DTZ001 - EXIF times are naive


def _frame(offset_s: float, body: float | None = None) -> Frame:
    row = dict.fromkeys(COLUMNS)
    row.update(bird_count=1 if body else 0, sharpness=body)
    frame = Frame(Path("x.jpg"), PhotoMeta(T0 + timedelta(seconds=offset_s), None, None), row)
    if body:
        frame.subject = Image.new("RGB", (8, 8))
        frame.bird = Bird((0, 0, 8, 8), np.ones((8, 8), dtype=bool))
    return frame


def test_bursts_splitOnTimeGapOnly():
    frames = [_frame(0.0), _frame(0.125), _frame(1.625), _frame(1.75)]
    assert [len(b) for b in _bursts(frames)] == [2, 2]
    assert [f.row["burst"] for f in frames] == [1, 1, 2, 2]


def test_pickBest_ranksBySharpnessAndKeepsOnlyBestCrop():
    # trace: sharpness a=50, b=70, c=60 → ranks b, c, a
    a, b, c = _frame(0.0, 50.0), _frame(0.125, 70.0), _frame(0.25, 60.0)
    assert _pick_best([a, b, c]) is b
    assert [f.row["burst_rank"] for f in (a, b, c)] == [3, 1, 2]
    assert (a.subject, c.subject) == (None, None) and b.subject is not None


def test_pickBest_noBird_returnsNone():
    assert _pick_best([_frame(0.0), _frame(0.125)]) is None
