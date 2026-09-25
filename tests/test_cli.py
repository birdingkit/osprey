from pathlib import Path

from PIL import ExifTags, Image

from osprey.cli import Shot, _bursts, _capture_time, _rename, _shots


def _touch(*paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def _photo(path: Path, taken: str | None = None, subsec: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    ifd = exif.get_ifd(ExifTags.IFD.Exif)
    if taken:
        ifd[ExifTags.Base.DateTimeOriginal] = taken
    if subsec:
        ifd[ExifTags.Base.SubsecTimeOriginal] = subsec
    Image.new("RGB", (8, 8)).save(path, exif=exif)
    return path


def _shot(name: str, taken: float | None, folder: Path = Path("day1")) -> Shot:
    return Shot(folder / name, [folder / name], taken)


def test_shots_groupsSameNameFilesAndSkipsRenamedAndHidden(tmp_path):
    _photo(tmp_path / "DSC0001.JPG")
    _photo(tmp_path / "day2" / "DSC0002.jpg")
    _photo(tmp_path / ".thumbnails" / "DSC0006.JPG")  # hidden folder
    _photo(tmp_path / "0003-01_DSC0005.JPG")  # renamed on an earlier run
    _touch(
        tmp_path / "DSC0001.ARW",
        tmp_path / "DSC0001.ARW.xmp",
        tmp_path / "DSC0003.ARW",  # RAW without a photo: not judged
        tmp_path / "._DSC0004.JPG",  # macOS resource fork
        tmp_path / "0003-01_DSC0005.ARW",
    )
    shots, _ = _shots(tmp_path)
    assert [s.photo.relative_to(tmp_path) for s in shots] == [Path("DSC0001.JPG"), Path("day2/DSC0002.jpg")]
    assert sorted(f.name for f in shots[0].files) == ["DSC0001.ARW", "DSC0001.ARW.xmp", "DSC0001.JPG"]


def test_captureTime_addsSubseconds(tmp_path):
    whole = _capture_time(_photo(tmp_path / "a.jpg", "2026:08:22 07:40:38"))
    # trace: subsec "347" → +0.347 s
    assert _capture_time(_photo(tmp_path / "b.jpg", "2026:08:22 07:40:38", "347")) == whole + 0.347


def test_captureTime_missingOrBlankDate_isNone(tmp_path):
    assert _capture_time(_photo(tmp_path / "a.jpg")) is None
    assert _capture_time(_photo(tmp_path / "b.jpg", "0000:00:00 00:00:00")) is None


def test_bursts_splitAtGapFolderAndMissingTime():
    other = Path("day2")
    shots = [
        _shot("DSC0003.JPG", 100.9),  # 0.775 s after DSC0002: same burst
        _shot("DSC0001.JPG", 100.0),
        _shot("DSC0002.JPG", 100.125),
        _shot("DSC0004.JPG", 101.9),  # 1.0 s gap: new burst
        _shot("DSC0005.JPG", 102.0, other),  # close in time but another folder
        _shot("DSC0006.JPG", None),  # no capture time: alone
    ]
    assert [[s.photo.name for s in b] for b in _bursts(shots)] == [
        ["DSC0001.JPG", "DSC0002.JPG", "DSC0003.JPG"],
        ["DSC0004.JPG"],
        ["DSC0006.JPG"],
        ["DSC0005.JPG"],
    ]


def test_shots_firstBurst_continuesAfterEarlierRuns(tmp_path):
    assert _shots(tmp_path)[1] == 1
    _touch(tmp_path / "0009-02_DSC0001.JPG", tmp_path / "sub" / "0012-01_DSC0002.ARW")
    _photo(tmp_path / "DSC_0003.JPG")  # underscore but no burst prefix
    assert _shots(tmp_path)[1] == 13


def test_rename_takesSidecarsAndRefusesToOverwrite(tmp_path):
    photo, raw = tmp_path / "DSC0001.JPG", tmp_path / "DSC0001.ARW"
    _touch(photo, raw)
    assert _rename(Shot(photo, [photo, raw], None), "0001-01_")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["0001-01_DSC0001.ARW", "0001-01_DSC0001.JPG"]

    _touch(photo)
    assert not _rename(Shot(photo, [photo], None), "0001-01_")
    assert photo.exists()
