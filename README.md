# osprey

Sort a folder of bird photos into one CSV: species (like iNaturalist, using date + place) and a sharpness score, so the keepers float to the top.

```sh
uv run osprey examples --place Taiwan
# → examples/osprey.csv
```

```
[1/3] DSC06266.JPG: 白腹鰹鳥 100% · soft
[2/3] DSC07300.JPG: 白眉燕鷗 100% · sharp
[3/3] DSC07714.JPG: 穴鳥 100% · sharp
```

Rows are grouped by species, sharpest first. Open the CSV in Excel / Numbers (UTF-8 with BOM, so Chinese names display correctly).

## Options

| Flag | Meaning |
|---|---|
| `--place Taiwan` | Where photos **without GPS** were taken. iNaturalist place name or id (`宜蘭縣`, `7887`). Omit → every bird species worldwide. |
| `--radius 50` | km around **GPS-tagged** photos to look for species (default 50). GPS always wins over `--place`. |
| `-o out.csv` | Output path (default `<folder>/osprey.csv`). |

## CSV columns

| Column | Meaning |
|---|---|
| `species_zh` / `species_en` / `scientific` | Best guess (names from iNaturalist, zh-TW) |
| `confidence` | Probability among local species |
| `alt_2`, `alt_3` | Runner-up species — check these when confidence is low |
| `in_season` | Species recorded nearby within ±1 month of the photo date |
| `quality` | `sharp` (≥60) · `soft` (40–60) · `blurry` (<40) · `no bird` |
| `sharpness` | 0–100 score of the bird itself, background ignored |
| `bird_size_pct` | Bird area as % of frame — tiny birds crop poorly |
| `bird_count` | Birds detected in frame |

## How it works

1. **Find the bird** — Mask R-CNN (COCO `bird` class) gives a box and a pixel mask; the largest bird is the subject.
2. **Sharpness** — re-blur metric (Crete et al. 2007, as in `skimage.measure.blur_effect`) computed only on bird pixels, so bokeh backgrounds don't count against the photo. Contrast-invariant: a dark petrel over dark sea scores like a white egret.
3. **Local species list** — iNaturalist research-grade bird observations for the place (or GPS radius): every species ever recorded there is a candidate; species not recorded within ±1 month of the photo date get a 10× lower prior.
4. **Species ID** — [BioCLIP 2.5](https://huggingface.co/imageomics/bioclip-2.5-vith14) (ViT-H/14, trained on the 200M-image TreeOfLife dataset) scores the bird crop against those candidates, using the same label text BioCLIP was trained on (7-rank taxonomy + common name, 80 prompt templates).

Speed on an M4 Mac: ~0.75 s per 33 MP photo (1,000 photos ≈ 13 min).

First run downloads ~4 GB of model weights and fetches/encodes the local species list (a few minutes per new place). All cached under `~/.cache/osprey` and `~/.cache/huggingface`; iNaturalist lists refresh after 30 days.
