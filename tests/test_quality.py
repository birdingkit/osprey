import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from osprey.detect import Animal
from osprey.quality import animal_sharpness, sharpness


def _checkerboard(contrast: float) -> np.ndarray:
    tile = np.kron([[0, 1] * 8, [1, 0] * 8] * 8, np.ones((16, 16)))
    return 100 + contrast * tile


def test_sharpness_blurredImage_scoresLower():
    image = _checkerboard(contrast=100)
    blurred = gaussian_filter(image, sigma=2)
    assert sharpness(image) > 60
    assert sharpness(blurred) < 40


def test_sharpness_lowContrastAnimal_scoresSameAsHighContrast():
    assert abs(sharpness(_checkerboard(20)) - sharpness(_checkerboard(120))) < 1


def test_sharpness_mask_ignoresBlurryBackground():
    sharp = _checkerboard(contrast=100)
    image = gaussian_filter(sharp, sigma=4)
    mask = np.zeros(image.shape, dtype=bool)
    mask[:, :128] = True
    image[:, :128] = sharp[:, :128]
    assert sharpness(image, mask) > sharpness(image) + 10


def _animal(box, shape):
    return Animal(box=box, mask=np.ones(shape, dtype=bool))


def test_animalSharpness_largeAnimal_judgedAtMaxSide():
    # trace: 2048 px box → resized to 1024 before scoring; a checkerboard stays sharp
    pixels = np.kron(_checkerboard(contrast=100), np.ones((8, 8))).astype(np.uint8)
    image = Image.fromarray(pixels).convert("RGB")
    assert animal_sharpness(image, _animal((0, 0, 2048, 2048), (2048, 2048))) > 60
