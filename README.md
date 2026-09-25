# osprey

Sort a folder of bird photos by species and sharpness. Bursts are grouped, the sharpest frame of each is identified, and everything goes into one CSV.

## Install

Needs a Mac with Apple Silicon and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/birdingkit/osprey.git
cd osprey
uv sync
```

The first run downloads about 4 GB of models.

## Use

```sh
uv run osprey ~/Pictures/2026-08-22 --place Taiwan
```

```
[1/3] burst 1 (1 frames): Brown Booby 100% · best DSC06266.JPG soft · exposure ok
[2/3] burst 2 (1 frames): Bridled Tern 100% · best DSC07300.JPG sharp · exposure ok
[3/3] burst 3 (1 frames): Bulwer's Petrel 100% · best DSC07714.JPG sharp · exposure ok
```

Results are written to `osprey.csv` in the photo folder, sorted by species, then burst, best frame first.

`--place` tells osprey where photos without GPS were taken. It takes an iNaturalist place name or id, such as `Taiwan`, `Yilan` or `7887`. Photos with GPS use the area around their own coordinates instead.

| Option | Default | |
|---|---|---|
| `--place` | worldwide | Place for photos without GPS |
| `--radius` | 50 | Search radius in km around GPS-tagged photos |
| `-o` | `<folder>/osprey.csv` | Output file |

## Output

| Column | |
|---|---|
| `burst`, `burst_rank` | Frames shot within 1 s of each other share a burst. Rank 1 is the sharpest. |
| `species_en`, `species_zh`, `scientific` | Species, identified on the burst's best frame |
| `confidence`, `alt_2`, `alt_3` | Match probability and the two runners-up |
| `in_season` | Whether the species has been recorded nearby within a month of the photo date |
| `quality` | `sharp` (60+), `soft` (40–60), `blurry` (under 40) or `no bird` |
| `sharpness` | 0–100, measured on the bird only |
| `exposure` | `over` if more than 1% of the bird is blown out, `under` if more than 10% is pure black |
| `blown_pct`, `crushed_pct` | Share of bird pixels clipped to white or black |
| `bird_size_pct`, `bird_count` | Bird area as a share of the frame, and birds detected |

## How it works

Mask R-CNN finds the bird. Sharpness uses the re-blur metric of Crete et al. (2007) on the bird's pixels, so a soft background does not count against the photo. Species candidates come from iNaturalist research-grade records for the place, and [BioCLIP 2.5](https://huggingface.co/imageomics/bioclip-2.5-vith14) picks among them.

A photo takes about 0.4 s on an M4 Mac.
