"""Capture date and GPS position from EXIF."""

from dataclasses import dataclass
from datetime import datetime

from PIL import Image
from PIL.ExifTags import GPS, IFD, Base


@dataclass(frozen=True)
class PhotoMeta:
    taken_at: datetime | None
    lat: float | None
    lon: float | None


def read_meta(image: Image.Image) -> PhotoMeta:
    exif = image.getexif()
    raw = exif.get_ifd(IFD.Exif).get(Base.DateTimeOriginal) or exif.get(Base.DateTime)
    # Camera wall-clock time, kept naive on purpose: only the month feeds the season check.
    taken_at = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S") if raw else None  # noqa: DTZ007
    lat, lon = _gps(exif.get_ifd(IFD.GPSInfo))
    return PhotoMeta(taken_at, lat, lon)


def _gps(gps: dict) -> tuple[float | None, float | None]:
    tags = (GPS.GPSLatitudeRef, GPS.GPSLatitude, GPS.GPSLongitudeRef, GPS.GPSLongitude)
    if not all(tag in gps for tag in tags):
        return None, None
    lat = _degrees(gps[GPS.GPSLatitude]) * (-1 if gps[GPS.GPSLatitudeRef] == "S" else 1)
    lon = _degrees(gps[GPS.GPSLongitude]) * (-1 if gps[GPS.GPSLongitudeRef] == "W" else 1)
    return lat, lon


def _degrees(dms) -> float:
    d, m, s = (float(x) for x in dms)
    return d + m / 60 + s / 3600
