"""osprey: move wildlife photos into A_sharp / B_soft / C_blurry / D_no_animal folders by how sharp the animal is."""

import argparse
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import huggingface_hub
import torch
import transformers
from PIL import Image, ImageOps

from .detect import AnimalDetector
from .quality import LABELS, animal_sharpness, quality_label

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
NO_ANIMAL = "D_no_animal"
FOLDERS = (*LABELS, NO_ANIMAL)


@dataclass
class Shot:
    """One photo plus every file sharing its name (RAW, XMP), which move together."""

    photo: Path
    files: list[Path]


def main() -> None:
    args = _parse_args()
    # Public model weights need no token; silence the Hub's nag about it.
    huggingface_hub.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
    shots = _shots(args.folder)
    if not shots:
        sys.exit(f"No photos in {args.folder}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    detector = AnimalDetector(device)

    counts = Counter()
    start = time.time()
    for i, (shot, image, pixels) in enumerate(_load_ahead(shots, detector.shrink), 1):
        score, folder = "-", NO_ANIMAL
        if animal := detector(image, pixels):
            score = animal_sharpness(image, animal)
            folder = quality_label(score)
        moved = _move(shot, args.folder / folder)
        counts[folder] += moved
        status = f"→ {folder}/" if moved else f"skipped: {folder}/{shot.photo.name} exists"
        print(f"[{i}/{len(shots)}] {shot.photo.name} {score} {status}", flush=True)

    summary = ", ".join(f"{counts[f]} {f}" for f in FOLDERS)
    print(f"{len(shots)} photos in {time.time() - start:.1f}s: {summary}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="osprey", description=__doc__)
    p.add_argument("folder", type=Path, help="folder of photos (searched recursively)")
    return p.parse_args()


def _shots(folder: Path) -> list[Shot]:
    """Photos under `folder`, each grouped with its same-name files; already-sorted folders are skipped."""
    groups: dict[tuple[Path, str], list[Path]] = defaultdict(list)
    for root, dirs, files in folder.walk():
        # Skip hidden folders and, at the top, the folders earlier runs sorted into
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not (root == folder and d in FOLDERS))
        for name in sorted(files):
            if not name.startswith("."):  # ._DSC0001.JPG is a macOS resource fork
                # DSC0001.JPG, DSC0001.ARW and DSC0001.ARW.xmp share the key DSC0001
                groups[root, name.partition(".")[0]].append(root / name)
    shots = []
    for files in groups.values():
        if photo := next((f for f in files if f.suffix.lower() in PHOTO_SUFFIXES), None):
            shots.append(Shot(photo, files))
    return shots


def _move(shot: Shot, dest: Path) -> bool:
    """Move all of the shot's files into `dest`; False (nothing moved) if any name is already taken."""
    if any((dest / f.name).exists() for f in shot.files):
        return False
    dest.mkdir(exist_ok=True)
    for f in shot.files:
        f.rename(dest / f.name)
    return True


def _load_ahead(shots: list[Shot], shrink: Callable) -> Iterator[tuple[Shot, Image.Image, torch.Tensor]]:
    """Decode and shrink the next photo on a worker thread while the GPU handles the current one."""
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(_load, shots[0], shrink)
        for next_shot in [*shots[1:], None]:
            loaded = future.result()
            if next_shot:
                future = pool.submit(_load, next_shot, shrink)
            yield loaded


def _load(shot: Shot, shrink: Callable) -> tuple[Shot, Image.Image, torch.Tensor]:
    image = Image.open(shot.photo)
    image.load()  # decodes pixels and closes the file
    if image.mode != "RGB":
        image = image.convert("RGB")
    ImageOps.exif_transpose(image, in_place=True)
    return shot, image, shrink(image)
