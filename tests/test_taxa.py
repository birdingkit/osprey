from osprey.taxa import Region, Species, season_months


def test_seasonMonths_january_wrapsToDecember():
    assert season_months(1) == (12, 1, 2)
    assert season_months(12) == (11, 12, 1)
    assert season_months(8) == (7, 8, 9)


def test_clipText_matchesBioclipTrainingFormat():
    tern = Species(
        4,
        "Onychoprion anaethetus",
        "Bridled Tern",
        "白眉燕鷗",
        ("Animalia", "Chordata", "Aves", "Charadriiformes", "Laridae"),
    )
    assert tern.clip_text() == (
        "Animalia Chordata Aves Charadriiformes Laridae Onychoprion anaethetus with common name Bridled Tern"
    )


def test_regionParams_prefersPlaceOverPoint():
    assert Region(place_id=7887).params() == {"place_id": 7887}
    assert Region.around(25.0341, 121.5645, radius_km=30).params() == {"lat": 25.0, "lng": 121.6, "radius": 30}
    assert Region().params() == {}
