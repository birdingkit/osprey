import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from osprey.detect import Bird
from osprey.quality import bird_sharpness, quality_label, sharpness


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


def test_birdSharpness_largeBird_judgedAtMaxSide():
    # trace: 2048 px box → resized to 1024 before scoring; a checkerboard stays sharp
    pixels = np.kron(_checkerboard(contrast=100), np.ones((8, 8))).astype(np.uint8)
    image = Image.fromarray(pixels).convert("RGB")
    assert bird_sharpness(image, _bird((0, 0, 2048, 2048), (2048, 2048))) > 60
