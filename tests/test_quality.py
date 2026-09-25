import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from osprey.detect import Bird
from osprey.quality import clipped_pct, exposure_label, quality_label, sharpness


def _checkerboard(contrast: float) -> np.ndarray:
    tile = np.kron([[0, 1] * 8, [1, 0] * 8] * 8, np.ones((16, 16)))
    return 100 + contrast * tile


def test_sharpness_blurredImage_scoresLower():
    image = _checkerboard(contrast=100)
    blurred = gaussian_filter(image, sigma=2)
    assert sharpness(image) > 60
    assert sharpness(blurred) < 40


def test_sharpness_lowContrastBird_scoresSameAsHighContrast():
    assert abs(sharpness(_checkerboard(20)) - sharpness(_checkerboard(120))) < 1


def test_sharpness_mask_ignoresBlurryBackground():
    sharp = _checkerboard(contrast=100)
    image = gaussian_filter(sharp, sigma=4)
    mask = np.zeros(image.shape, dtype=bool)
    mask[:, :128] = True
    image[:, :128] = sharp[:, :128]
    assert sharpness(image, mask) > sharpness(image) + 10


def _bird(box, shape):
    return Bird(box=box, mask=np.ones(shape, dtype=bool))


def test_qualityLabel_cutoffs():
    # trace: sharp >= 60, soft >= 40
    assert quality_label(61.4) == "sharp"
    assert quality_label(58.2) == "soft"
    assert quality_label(30.0) == "blurry"


def test_exposureLabel_flagsClippedPlumage():
    assert exposure_label(1.6, 0.0) == "over"
    assert exposure_label(0.0, 16.1) == "under"
    assert exposure_label(0.0, 5.6) == "ok"
    assert exposure_label(1.0, 10.0) == "ok"


def test_clippedPct_countsBlownAndCrushedBirdPixels():
    pixels = np.full((10, 10, 3), 128, dtype=np.uint8)
    pixels[0, :] = 255  # 10 of 100 blown
    pixels[1:3, :] = 0  # 20 of 100 crushed
    assert clipped_pct(Image.fromarray(pixels), _bird((0, 0, 10, 10), (10, 10))) == (10.0, 20.0)
