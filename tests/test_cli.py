from pathlib import Path

from osprey.cli import Shot, _move, _shots


def _touch(*paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_shots_groupsSameNameFilesAndSkipsSortedFolders(tmp_path):
    _touch(
        tmp_path / "DSC0001.JPG",
        tmp_path / "DSC0001.ARW",
        tmp_path / "DSC0001.ARW.xmp",
        tmp_path / "day2" / "DSC0002.jpg",
        tmp_path / "DSC0003.ARW",  # RAW without a photo: not judged
        tmp_path / "._DSC0004.JPG",  # macOS resource fork
        tmp_path / ".thumbnails" / "DSC0006.JPG",  # hidden folder
        tmp_path / "day2" / "A_sharp" / "DSC0007.JPG",  # only top-level sorted folders are skipped
        tmp_path / "A_sharp" / "DSC0005.JPG",  # sorted on an earlier run
    )
    shots = _shots(tmp_path)
    assert [s.photo.relative_to(tmp_path) for s in shots] == [
        Path("DSC0001.JPG"),
        Path("day2/DSC0002.jpg"),
        Path("day2/A_sharp/DSC0007.JPG"),
    ]
    assert sorted(f.name for f in shots[0].files) == ["DSC0001.ARW", "DSC0001.ARW.xmp", "DSC0001.JPG"]


def test_move_takesSidecarsAndRefusesToOverwrite(tmp_path):
    photo, raw = tmp_path / "DSC0001.JPG", tmp_path / "DSC0001.ARW"
    _touch(photo, raw)
    assert _move(Shot(photo, [photo, raw]), tmp_path / "A_sharp")
    assert sorted(p.name for p in (tmp_path / "A_sharp").iterdir()) == ["DSC0001.ARW", "DSC0001.JPG"]

    _touch(photo)
    assert not _move(Shot(photo, [photo]), tmp_path / "A_sharp")
    assert photo.exists()
