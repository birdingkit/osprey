"""osprey: sort bird photos by species and sharpness into one CSV."""

import argparse
import csv
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import batched
from pathlib import Path

import torch
from PIL import Image, ImageOps

from .detect import BirdDetector
from .exif import PhotoMeta, read_meta
from .quality import bird_sharpness, quality_label
from .species import Guess, SpeciesClassifier, species_crop
from .taxa import DEFAULT_RADIUS_KM, Region, resolve_place, species_near

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
BATCH = 16  # photos per species-classifier batch

COLUMNS = [
    "file",
    "taken_at",
    "species_zh",
    "species_en",
    "scientific",
    "confidence",
    "alt_2",
    "alt_3",
    "in_season",
    "quality",
    "sharpness",
    "bird_size_pct",
    "bird_count",
    "lat",
    "lon",
]


@dataclass
class Photo:
    path: Path
    meta: PhotoMeta
    image: Image.Image


@dataclass
class Pending:
    """A photo analyzed for quality, waiting for a batched species ID."""

    path: Path
    meta: PhotoMeta
    row: dict
    crop: Image.Image | None  # None when no bird was found


def main() -> None:
    args = _parse_args()
    paths = sorted(p for p in args.folder.rglob("*") if p.suffix.lower() in PHOTO_SUFFIXES)
    if not paths:
        sys.exit(f"No photos in {args.folder}")

    fallback_region = Region()
    if args.place:
        place_id, name = resolve_place(args.place)
        print(f"Place: {name} (iNaturalist place {place_id})")
        fallback_region = Region(place_id=place_id)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    detector = BirdDetector(device)
    classifier = SpeciesClassifier(device)

    rows = []
    start = time.time()
    for batch in batched((_analyze(photo, detector) for photo in _load_ahead(paths)), BATCH):
        with_bird = [p for p in batch if p.crop is not None]
        if with_bird:
            for p, embedding in zip(with_bird, classifier.embed([p.crop for p in with_bird])):
                region = (
                    Region.around(p.meta.lat, p.meta.lon, args.radius) if p.meta.lat is not None else fallback_region
                )
                month = p.meta.taken_at.month if p.meta.taken_at else None
                _fill_species(p.row, classifier.rank(embedding, *species_near(region, month)))
        for p in batch:
            rows.append(p.row)
            species = p.row["species_zh"] or p.row["scientific"] or "-"
            print(
                f"[{len(rows)}/{len(paths)}] {p.path.name}: {species} {p.row['confidence'] or ''} · {p.row['quality']}"
            )

    rows.sort(key=lambda r: (r["scientific"] is None, r["scientific"] or "", -(r["sharpness"] or 0)))
    out = args.output or args.folder / "osprey.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:  # BOM so Excel reads Chinese
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} photos in {time.time() - start:.0f}s → {out}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="osprey", description=__doc__)
    p.add_argument("folder", type=Path, help="folder of photos (searched recursively)")
    p.add_argument(
        "--place", help="where photos without GPS were taken: iNaturalist place name or id, e.g. Taiwan, 宜蘭縣"
    )
    p.add_argument(
        "--radius",
        type=float,
        default=DEFAULT_RADIUS_KM,
        help=f"km around GPS-tagged photos to look for species (default {DEFAULT_RADIUS_KM})",
    )
    p.add_argument("-o", "--output", type=Path, help="CSV path (default: <folder>/osprey.csv)")
    return p.parse_args()


def _load_ahead(paths: list[Path]) -> Iterator[Photo]:
    """Decode the next JPEG on a worker thread while the GPU handles the current one."""
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(_load, paths[0])
        for next_path in [*paths[1:], None]:
            photo = future.result()
            if next_path:
                future = pool.submit(_load, next_path)
            yield photo


def _load(path: Path) -> Photo:
    image = Image.open(path)
    meta = read_meta(image)
    image.load()  # decodes pixels and closes the file
    if image.mode != "RGB":
        image = image.convert("RGB")
    ImageOps.exif_transpose(image, in_place=True)
    return Photo(path, meta, image)


def _analyze(photo: Photo, detector: BirdDetector) -> Pending:
    """CSV row with quality columns filled, plus a small crop of the bird to identify."""
    meta, image = photo.meta, photo.image
    row = dict.fromkeys(COLUMNS)
    row.update(
        file=str(photo.path),
        taken_at=meta.taken_at.isoformat(sep=" ") if meta.taken_at else None,
        lat=meta.lat,
        lon=meta.lon,
    )
    birds = detector(image)
    row["bird_count"] = len(birds)
    if not birds:
        row["quality"] = "no bird"
        return Pending(photo.path, meta, row, None)
    bird = birds[0]
    score = bird_sharpness(image, bird)
    row.update(
        sharpness=score,
        quality=quality_label(score),
        bird_size_pct=round(100 * bird.area / (image.width * image.height), 2),
    )
    return Pending(photo.path, meta, row, species_crop(image, bird))


def _fill_species(row: dict, guesses: list[Guess]) -> None:
    best = guesses[0]
    row.update(
        species_zh=best.species.common_zh,
        species_en=best.species.common_en,
        scientific=best.species.scientific,
        confidence=f"{best.probability:.0%}",
        in_season="yes" if best.in_season else "no",
    )
    for column, guess in zip(("alt_2", "alt_3"), guesses[1:]):
        row[column] = f"{guess.species.common_zh or guess.species.scientific} {guess.probability:.0%}"
