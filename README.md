# osprey

Sort a folder of wildlife photos (birds, dolphins, whales) by how sharp the animal is. osprey splits the photos into bursts and renames them in place, so each burst sits together with its sharpest frame first. RAW and XMP files are renamed along with their photo.

## Install

Needs a Mac with Apple Silicon and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/birdingkit/osprey.git
cd osprey
uv sync
```

The first run downloads the detector and segmenter (about 800 MB).

## Use

```sh
uv run osprey ~/Pictures/2026-08-22
```

```
[1/5] DSC05430.JPG 72.5 → 0001-01_DSC05430.JPG
[2/5] DSC05431.JPG 68.5 → 0001-02_DSC05431.JPG
[3/5] DSC05429.JPG 66.7 → 0001-03_DSC05429.JPG
[4/5] DSC05432.JPG 43.8 → 0001-04_DSC05432.JPG
[5/5] DSC05433.JPG 38.8 → 0001-05_DSC05433.JPG
5 photos in 1 bursts in 2.5s
```

`0001-03_DSC05429.JPG` is burst 1, third sharpest. Sorted by name, each burst sits together with its best frame first, so you can flip through a day without jumping between folders. The number is animal sharpness (0–100); `-` means no bird, dolphin or whale was found, and those frames go last in their burst.

Frames less than 1 s apart (by EXIF capture time) form one burst. A burst never spans subfolders, and a photo without a capture date is a burst of its own.

Photos already renamed are left alone, so running again only sorts new photos, numbered after the earlier bursts. A photo whose new name is already taken is skipped.

Files sharing the photo's name are renamed with it: `DSC0001.JPG`, `DSC0001.ARW` and `DSC0001.ARW.xmp` stay together.

osprey renames files, it never moves or deletes them. To undo, strip the prefix:

```sh
for f in **/[0-9][0-9][0-9][0-9]*-[0-9][0-9]*_*(N); do mv "$f" "${f:h}/${${f:t}#*_}"; done  # zsh
```

## How it works

[OWL-ViT](https://huggingface.co/google/owlvit-base-patch32) finds birds, dolphins and whales from text prompts, and [SAM 2](https://huggingface.co/facebook/sam2.1-hiera-tiny) outlines the largest one. Sharpness uses the re-blur metric of Crete et al. (2007) on the animal's pixels, so a soft background does not count against the photo.

A photo takes about 0.5 s on an M4 Mac.

## Limits

- Only birds, dolphins and whales are looked for. Add more names to `ANIMALS` in `src/osprey/detect.py`.
- Very faint or distant animals, such as a pale seabird soaring against grey sky, can be missed and get `-`. Check the last frames of a burst before deleting anything.
- The 1 s burst gap was set on Sony A7 IV bursts (8 fps). Change `BURST_GAP` in `src/osprey/cli.py` for other cameras or shooting styles.
