# osprey

Group a folder of photos into bursts. osprey renames each photo in place with its burst number, so the frames of one burst sit together and you can flip through a day burst by burst. RAW and XMP files are renamed along with their photo.

## Install

Needs [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/birdingkit/osprey.git
cd osprey
uv sync
```

## Use

```sh
uv run osprey ~/Pictures/2026-08-22
```

```
[1/5] DSC05409.JPG → 0001_DSC05409.JPG
[2/5] DSC05410.JPG → 0001_DSC05410.JPG
[3/5] DSC05418.JPG → 0002_DSC05418.JPG
[4/5] DSC05419.JPG → 0002_DSC05419.JPG
[5/5] DSC05420.JPG → 0002_DSC05420.JPG
5 photos in 2 bursts
```

`0002_DSC05419.JPG` is in burst 2. Sorted by name, bursts come in shooting order and frames keep their capture order inside a burst.

Frames less than 1 s apart (by EXIF capture time) form one burst. A burst never spans subfolders, and a photo without a capture date is a burst of its own.

Photos already renamed are left alone, so running again only sorts new photos, numbered after the earlier bursts. A photo whose new name is already taken is skipped.

Files sharing the photo's name are renamed with it: `DSC0001.JPG`, `DSC0001.ARW` and `DSC0001.ARW.xmp` stay together.

osprey renames files, it never moves or deletes them. To undo, strip the prefix:

```sh
for f in **/[0-9][0-9][0-9][0-9]_*(N); do mv "$f" "${f:h}/${${f:t}#*_}"; done  # zsh
```

A day of 1414 photos takes about a second: only the EXIF header is read.

## Limits

- The 1 s burst gap was set on Sony A7 IV bursts (8 fps). Change `BURST_GAP` in `src/osprey/cli.py` for other cameras or shooting styles.
- Burst numbers have four digits, so one folder holds up to 9999 bursts.
