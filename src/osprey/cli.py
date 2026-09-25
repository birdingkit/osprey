"""osprey: sort bird photos by species and sharpness into one CSV."""

import argparse
import csv
import logging
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import batched
from pathlib import Path

import torch
from PIL import Image, ImageOps

from .detect import Bird, BirdDetector
from .exif import PhotoMeta, read_meta
from .quality import bird_sharpness, clipped_pct, exposure_label, quality_label
from .species import Guess, SpeciesClassifier, context_box, species_crop
from .taxa import DEFAULT_RADIUS_KM, Region, resolve_place, species_near

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
BATCH = 16  # bursts per species-classifier batch
BURST_GAP_S = 1.0  # consecutive frames this close form a burst (8 fps bursts are 125 ms apart)

COLUMNS = [
    "file",
    "taken_at",
    "burst",
    "burst_rank",
    "species_en",
    "species_zh",
    "scientific",
    "confidence",
    "alt_2",
    "alt_3",
    "in_season",
    "quality",
    "exposure",
    "sharpness",
    "blown_pct",
    "crushed_pct",
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
    detector_input: tuple[torch.Tensor, float]


@dataclass
class Frame:
    """One photo after the cheap pass. Only the bird's surroundings are kept, not the full image."""

    path: Path
    meta: PhotoMeta
    row: dict
    subject: Image.Image | None = None  # full-resolution square around the bird
    bird: Bird | None = None  # in `subject` coordinates


def main() -> None:
    args = _parse_args()
    # Public model weights need no token; drop only the Hub's X-HF-Warning nag about it.
    logging.getLogger("huggingface_hub.utils._http").addFilter(
        lambda record: "unauthenticated requests" not in record.getMessage()
    )
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
    frames = (_analyze(photo, detector) for photo in _load_ahead(paths, detector.shrink))
    for bursts in batched(_bursts(frames), BATCH):
        # Species ID only for each burst's sharpest frame; its species covers the burst.
        picks = [(burst, best) for burst in bursts if (best := _pick_best(burst))]
        if picks:
            crops = [species_crop(best.subject, best.bird) for _, best in picks]
            for (burst, best), embedding in zip(picks, classifier.embed(crops)):
                region = (
                    Region.around(best.meta.lat, best.meta.lon, args.radius)
                    if best.meta.lat is not None
                    else fallback_region
                )
                month = best.meta.taken_at.month if best.meta.taken_at else None
                guesses = classifier.rank(embedding, *species_near(region, month))
                for frame in burst:
                    _fill_species(frame.row, guesses)
        best_of = {id(burst): best for burst, best in picks}
        for burst in bursts:
            rows.extend(frame.row for frame in burst)
            best = best_of.get(id(burst))
            if best is None:
                print(f"[{len(rows)}/{len(paths)}] burst {burst[0].row['burst']} ({len(burst)} frames): no bird")
                continue
            best.subject = None
            species = best.row["species_en"] or best.row["scientific"] or "-"
            print(
                f"[{len(rows)}/{len(paths)}] burst {best.row['burst']} ({len(burst)} frames): {species}"
                f" {best.row['confidence']} · best {best.path.name} {best.row['quality']}"
                f" · exposure {best.row['exposure']}",
                flush=True,
            )

    rows.sort(key=lambda r: (r["scientific"] is None, r["scientific"] or "", r["burst"], r["burst_rank"] or 999))
    out = args.output or args.folder / "osprey.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:  # BOM so Excel reads Chinese
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} photos in {time.time() - start:.1f}s → {out}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="osprey", description=__doc__)
    p.add_argument("folder", type=Path, help="folder of photos (searched recursively)")
    p.add_argument(
        "--place", help="where photos without GPS were taken: iNaturalist place name or id, e.g. Taiwan, Yilan"
    )
    p.add_argument(
        "--radius",
        type=float,
        default=DEFAULT_RADIUS_KM,
        help=f"km around GPS-tagged photos to look for species (default {DEFAULT_RADIUS_KM})",
    )
    p.add_argument("-o", "--output", type=Path, help="CSV path (default: <folder>/osprey.csv)")
    return p.parse_args()


def _load_ahead(paths: list[Path], shrink: Callable) -> Iterator[Photo]:
    """Decode and shrink the next photo on a worker thread while the GPU handles the current one."""
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(_load, paths[0], shrink)
        for next_path in [*paths[1:], None]:
            photo = future.result()
            if next_path:
                future = pool.submit(_load, next_path, shrink)
            yield photo


def _load(path: Path, shrink: Callable) -> Photo:
    image = Image.open(path)
    meta = read_meta(image)
    image.load()  # decodes pixels and closes the file
    if image.mode != "RGB":
        image = image.convert("RGB")
    ImageOps.exif_transpose(image, in_place=True)
    return Photo(path, meta, image, shrink(image))


def _analyze(photo: Photo, detector: BirdDetector) -> Frame:
    """Cheap pass for every photo: find the bird, score body sharpness and exposure."""
    meta, image = photo.meta, photo.image
    row = dict.fromkeys(COLUMNS)
    row.update(
        file=str(photo.path),
        taken_at=meta.taken_at.isoformat(sep=" ", timespec="milliseconds") if meta.taken_at else None,
        lat=meta.lat,
        lon=meta.lon,
    )
    birds = detector(photo.detector_input)
    row["bird_count"] = len(birds)
    if not birds:
        row["quality"] = "no bird"
        return Frame(photo.path, meta, row)
    crop_box = context_box(birds[0], image.size)
    subject, bird = image.crop(crop_box), birds[0].within(crop_box)
    score = bird_sharpness(subject, bird)
    blown, crushed = clipped_pct(subject, bird)
    row.update(
        quality=quality_label(score),
        exposure=exposure_label(blown, crushed),
        sharpness=score,
        blown_pct=blown,
        crushed_pct=crushed,
        bird_size_pct=round(100 * bird.area / (image.width * image.height), 2),
    )
    return Frame(photo.path, meta, row, subject, bird)


def _bursts(frames: Iterable[Frame]) -> Iterator[list[Frame]]:
    """Group consecutive frames shot <= BURST_GAP_S apart; every row gets its burst number."""
    burst: list[Frame] = []
    burst_id = 1
    for frame in frames:
        if burst and not _within_gap(burst[-1], frame):
            yield burst
            burst = []
            burst_id += 1
        frame.row["burst"] = burst_id
        burst.append(frame)
    if burst:
        yield burst


def _within_gap(previous: Frame, frame: Frame) -> bool:
    if previous.meta.taken_at is None or frame.meta.taken_at is None:
        return False
    return (frame.meta.taken_at - previous.meta.taken_at).total_seconds() <= BURST_GAP_S


def _pick_best(burst: list[Frame]) -> Frame | None:
    """Rank the burst by bird sharpness and return its best frame (None if no bird)."""
    ranked = sorted((f for f in burst if f.bird), key=lambda f: f.row["sharpness"], reverse=True)
    for rank, frame in enumerate(ranked, 1):
        frame.row["burst_rank"] = rank
        if rank > 1:
            frame.subject = None  # only the best frame is identified
    return ranked[0] if ranked else None


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
