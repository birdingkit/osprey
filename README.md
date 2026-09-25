# osprey

Sort a folder of wildlife photos (birds, dolphins, whales) by how sharp the animal is. Each photo moves into `A_sharp/`, `B_soft/`, `C_blurry/` or `D_no_animal/` inside the folder, together with its RAW and XMP files.

## Install

Needs a Mac with Apple Silicon and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/birdingkit/osprey.git
cd osprey
uv sync
```

The first run downloads the detector and segmenter (about 1 GB).

## Use

```sh
uv run osprey ~/Pictures/2026-08-22
```

```
[1/3] DSC06266.JPG 59.8 → B_soft/
[2/3] DSC07300.JPG 70.1 → A_sharp/
[3/3] DSC07714.JPG 64.2 → A_sharp/
3 photos in 8.5s: 2 A_sharp, 1 B_soft, 0 C_blurry, 0 D_no_animal
```

Subfolders are searched too, but everything lands in the four folders at the top. Photos already in those folders are left alone, so running again only sorts new photos. A photo whose name is already taken in its target folder is skipped.

| Folder | Animal sharpness (0–100) |
|---|---|
| `A_sharp` | 60+ |
| `B_soft` | 40–60 |
| `C_blurry` | under 40 |
| `D_no_animal` | no bird, dolphin or whale found |

The letter prefix keeps the folders in best-first order in Finder.

Files sharing the photo's name move with it: `DSC0001.JPG`, `DSC0001.ARW` and `DSC0001.ARW.xmp` stay together.

## How it works

[OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble) finds birds, dolphins and whales from text prompts, and [SAM](https://huggingface.co/facebook/sam-vit-base) outlines the largest one. Sharpness uses the re-blur metric of Crete et al. (2007) on the animal's pixels, so a soft background does not count against the photo.

A photo takes about 2.7 s on an M4 Mac.
