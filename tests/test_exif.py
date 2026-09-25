from osprey.exif import _gps


def test_gps_southWest_isNegative():
    gps = {1: "S", 2: (33.0, 51.0, 36.0), 3: "W", 4: (151.0, 12.0, 0.0)}
    assert _gps(gps) == (-33.86, -151.2)
    assert _gps({}) == (None, None)
