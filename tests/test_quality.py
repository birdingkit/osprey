import numpy as np
from scipy.ndimage import gaussian_filter

from osprey.quality import sharpness


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
