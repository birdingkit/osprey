"""osprey: rename photos so each burst of frames sits together."""

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from PIL import ExifTags, Image

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
# Frames closer than this are one burst. Sony A7 IV bursts run 8 fps (0.125 s); re-pressing
# the shutter on the same scene leaves 0.5-2 s gaps, and 1 s splits those best on real outings.
BURST_GAP = 1.0
# 0012_DSC07300.JPG: burst 12. Photos named like this were sorted on an earlier run.
SORTED = re.compile(r"^(\d{4})_")


@dataclass
class Shot:
    """One photo plus every file sharing its name (RAW, XMP), which are renamed together."""

    photo: Path
    files: list[Path]
    taken: float | None  # capture time in seconds, None if the photo has no EXIF date


def main() -> None:
    args = _parse_args()
    shots, first_burst = _shots(args.folder)
    bursts = _bursts(shots)
    if not bursts:
        sys.exit(f"No unsorted photos in {args.folder}")

    done = 0
    for number, burst in enumerate(bursts, first_burst):
        prefix = f"{number:04d}_"
        for shot in burst:
            done += 1
            status = f"→ {prefix}{shot.photo.name}" if _rename(shot, prefix) else "skipped: name taken"
            print(f"[{done}/{len(shots)}] {shot.photo.name} {status}")

    print(f"{len(shots)} photos in {len(bursts)} bursts")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="osprey", description=__doc__)
    p.add_argument("folder", type=Path, help="folder of photos (searched recursively)")
    return p.parse_args()


def _shots(folder: Path) -> tuple[list[Shot], int]:
    """Unsorted photos under `folder`, each grouped with its same-name files, and the first burst number
    after those used by earlier runs, so new bursts sort after them."""
    groups: dict[tuple[Path, str], list[Path]] = defaultdict(list)
    last_burst = 0
    for root, dirs, files in folder.walk():
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            if name.startswith("."):  # ._DSC0001.JPG is a macOS resource fork
                continue
            if sorted_name := SORTED.match(name):
                last_burst = max(last_burst, int(sorted_name[1]))
            else:
                # DSC0001.JPG, DSC0001.ARW and DSC0001.ARW.xmp share the key DSC0001
                groups[root, name.partition(".")[0]].append(root / name)
    shots = []
    for files in groups.values():
        if photo := next((f for f in files if f.suffix.lower() in PHOTO_SUFFIXES), None):
            shots.append(Shot(photo, files, _capture_time(photo)))
    return shots, last_burst + 1


def _capture_time(photo: Path) -> float | None:
    with Image.open(photo) as image:
        exif = image.getexif().get_ifd(ExifTags.IFD.Exif)
    subsec = str(exif.get(ExifTags.Base.SubsecTimeOriginal, "")).strip()
    try:
        # Camera clock has no zone; only gaps between frames matter, so read it as UTC
        taken = datetime.strptime(exif[ExifTags.Base.DateTimeOriginal], "%Y:%m:%d %H:%M:%S").replace(tzinfo=UTC)
    except (KeyError, ValueError):  # no date, or a blank one like 0000:00:00 00:00:00
        return None
    return taken.timestamp() + float(f"0.{subsec or 0}")


def _bursts(shots: list[Shot]) -> list[list[Shot]]:
    """Shots in capture order, split wherever the gap reaches BURST_GAP; a burst never spans folders."""
    shots = sorted(shots, key=lambda s: (s.photo.parent, s.taken is None, s.taken or 0, s.photo.name))
    bursts: list[list[Shot]] = []
    for previous, shot in zip([None, *shots], shots):
        same_burst = (
            previous is not None
            and previous.photo.parent == shot.photo.parent
            and previous.taken is not None
            and shot.taken is not None
            and shot.taken - previous.taken < BURST_GAP
        )
        if same_burst:
            bursts[-1].append(shot)
        else:
            bursts.append([shot])
    return bursts


def _rename(shot: Shot, prefix: str) -> bool:
    """Put `prefix` in front of all the shot's file names; False (nothing renamed) if any new name is taken."""
    if any(f.with_name(prefix + f.name).exists() for f in shot.files):
        return False
    for f in shot.files:
        f.rename(f.with_name(prefix + f.name))
    return True
