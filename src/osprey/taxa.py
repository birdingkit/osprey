"""Bird species recorded near a place, from iNaturalist research-grade observations.

This is the "seen nearby" signal iNaturalist uses: the candidate list is every
bird species observed in the region, and species also observed within one month
of the photo date count as in season.
"""

import functools
import hashlib
import json
import math
import time
from dataclasses import dataclass
from itertools import batched
from pathlib import Path

import requests

API = "https://api.inaturalist.org/v1"
CACHE_ROOT = Path.home() / ".cache" / "osprey"
CACHE_DIR = CACHE_ROOT / "inat"
CACHE_TTL_S = 30 * 24 * 3600
PER_PAGE = 500
IDS_PER_REQUEST = 30  # /taxa/{ids} limit
DEFAULT_RADIUS_KM = 50


@dataclass(frozen=True)
class Species:
    taxon_id: int
    scientific: str
    common_en: str
    common_zh: str
    lineage: tuple[str, ...]  # kingdom, phylum, class, order, family names

    def clip_text(self) -> str:
        """The label format BioCLIP was trained on: 7-rank taxonomy + common name."""
        text = " ".join((*self.lineage, self.scientific))
        return f"{text} with common name {self.common_en}" if self.common_en else text


@dataclass(frozen=True)
class Region:
    """A named iNaturalist place, or a circle around a point, or the whole world."""

    place_id: int | None = None
    lat: float | None = None
    lon: float | None = None
    radius_km: float = DEFAULT_RADIUS_KM

    @classmethod
    def around(cls, lat: float, lon: float, radius_km: float) -> "Region":
        """Circle around a GPS fix, snapped to a 0.1° (~10 km) grid so nearby photos share one lookup."""
        return cls(lat=round(lat, 1), lon=round(lon, 1), radius_km=radius_km)

    def params(self) -> dict:
        if self.place_id is not None:
            return {"place_id": self.place_id}
        if self.lat is not None:
            return {"lat": self.lat, "lng": self.lon, "radius": self.radius_km}
        return {}


def resolve_place(query: str) -> tuple[int, str]:
    """Place id + display name for an iNaturalist place id or search text ("Taiwan", "宜蘭")."""
    data = _get(f"/places/{query}", {}) if query.isdigit() else _get("/places/autocomplete", {"q": query})
    if not data["results"]:
        raise SystemExit(f"iNaturalist has no place matching {query!r}")
    place = data["results"][0]
    return place["id"], place["display_name"]


@functools.cache
def species_near(region: Region, month: int | None) -> tuple[list[Species], set[int]]:
    """All bird species recorded in the region, and the ids of those in season for `month`."""
    species = _species_in(region)
    if month is None:
        return species, {s.taxon_id for s in species}
    window = ",".join(map(str, season_months(month)))
    in_season = {t["id"] for t in _species_taxa({**region.params(), "month": window})}
    return species, in_season


def season_months(month: int) -> tuple[int, int, int]:
    """The photo month and its neighbours, wrapping around the year."""
    return tuple((month + delta - 1) % 12 + 1 for delta in (-1, 0, 1))


@functools.cache
def _species_in(region: Region) -> list[Species]:
    taxa = _species_taxa(region.params())
    # Genus is the species' own name prefix; skipping it keeps the lookup small.
    names = _rank_names({a for t in taxa for a in t["ancestor_ids"][:-2]})
    return [
        Species(
            taxon_id=t["id"],
            scientific=t["name"],
            common_en=t.get("english_common_name") or "",
            common_zh=t.get("preferred_common_name") or "",
            lineage=tuple(names[a] for a in t["ancestor_ids"] if a in names),
        )
        for t in taxa
    ]


def _species_taxa(params: dict) -> list[dict]:
    base = {
        **params,
        "iconic_taxa": "Aves",
        "quality_grade": "research",
        "locale": "zh-TW",
        "per_page": PER_PAGE,
    }
    first = _get("/observations/species_counts", {**base, "page": 1})
    results = first["results"]
    for page in range(2, math.ceil(first["total_results"] / PER_PAGE) + 1):
        results += _get("/observations/species_counts", {**base, "page": page})["results"]
    return [r["taxon"] for r in results if r["taxon"]["rank"] == "species"]


def _rank_names(taxon_ids: set[int]) -> dict[int, str]:
    """Names of the Linnaean-rank taxa (kingdom..family) among `taxon_ids`."""
    ranks = {"kingdom", "phylum", "class", "order", "family"}
    names = {}
    for chunk in batched(sorted(taxon_ids), IDS_PER_REQUEST):
        for t in _get(f"/taxa/{','.join(map(str, chunk))}", {})["results"]:
            if t["rank"] in ranks:
                names[t["id"]] = t["name"]
    return names


def _get(path: str, params: dict) -> dict:
    key = hashlib.sha1(json.dumps([path, params], sort_keys=True).encode()).hexdigest()
    cached = CACHE_DIR / f"{key}.json"
    if cached.exists() and time.time() - cached.stat().st_mtime < CACHE_TTL_S:
        return json.loads(cached.read_text())
    resp = requests.get(API + path, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(1)  # iNaturalist asks for <= 1 request/second
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached.write_text(resp.text)
    return resp.json()
