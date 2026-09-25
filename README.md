# osprey

Sort a folder of bird photos by how sharp the bird is. Each photo moves into `sharp/`, `soft/`, `blurry/` or `no_bird/` inside the folder, together with its RAW and XMP files.

## Install

Needs a Mac with Apple Silicon and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/birdingkit/osprey.git
cd osprey
uv sync
```

The first run downloads the bird detector (about 200 MB).

## Use

```sh
uv run osprey ~/Pictures/2026-08-22
```

```
[1/3] DSC06266.JPG 58.2 → soft/
[2/3] DSC07300.JPG 69.5 → sharp/
[3/3] DSC07714.JPG 62.7 → sharp/
3 photos in 1.6s: 2 sharp, 1 soft, 0 blurry, 0 no_bird
```

Subfolders are searched too, but everything lands in the four folders at the top. Photos already in those folders are left alone, so running again only sorts new photos. A photo whose name is already taken in its target folder is skipped.

| Folder | Bird sharpness (0–100) |
|---|---|
| `sharp` | 60+ |
| `soft` | 40–60 |
| `blurry` | under 40 |
| `no_bird` | no bird found |

Files sharing the photo's name move with it: `DSC0001.JPG`, `DSC0001.ARW` and `DSC0001.ARW.xmp` stay together.

## How it works

Mask R-CNN finds the bird. Sharpness uses the re-blur metric of Crete et al. (2007) on the bird's pixels, so a soft background does not count against the photo.
