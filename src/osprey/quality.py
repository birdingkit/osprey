"""Photo sharpness judged on the bird itself."""

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter1d

from .detect import Bird

MAX_SIDE = 1024  # judge at screen-viewing scale, never upscale
BLUR_WINDOW = 11
# Label cut-offs. Calibrated on in-focus Sony A7 IV frames (58-70) vs the same
# frames with Gaussian sigma=2 or 15 px motion blur (19-38).
SHARP, SOFT = 60.0, 40.0
LABELS = ("A_sharp", "B_soft", "C_blurry")  # letter prefix sorts folders best-first


def bird_sharpness(image: Image.Image, bird: Bird) -> float:
    """Sharpness of the bird's own pixels, judged at most MAX_SIDE across."""
    crop, mask = image.crop(bird.box).convert("L"), bird.mask
    scale = min(1.0, MAX_SIDE / max(crop.size))
    if scale < 1:
        size = (round(crop.width * scale), round(crop.height * scale))
        crop = crop.resize(size, Image.LANCZOS)
        mask = np.asarray(Image.fromarray(mask).resize(size, Image.NEAREST))
    return sharpness(np.asarray(crop), mask)


def quality_label(score: float) -> str:
    sharp, soft, blurry = LABELS
    return sharp if score >= SHARP else soft if score >= SOFT else blurry


def sharpness(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    """0-100, higher is sharper. `gray` is the bird crop, `mask` marks bird pixels.

    Re-blur metric (Crete et al. 2007, the one behind skimage.measure.blur_effect):
    blur the crop again and measure how much edge contrast is lost. Sharp edges
    lose a lot, already-blurry edges lose little. Contrast-invariant, so dark
    birds against the sky are scored the same as bright ones.
    """
    gray = gray.astype(np.float32)
    losses = []
    for axis in (0, 1):
        reblurred = uniform_filter1d(gray, BLUR_WINDOW, axis=axis)
        d_orig = np.abs(np.diff(gray, axis=axis))
        d_blur = np.abs(np.diff(reblurred, axis=axis))
        lost = np.maximum(0, d_orig - d_blur)
        if mask is not None:
            m = mask[1:, :] if axis == 0 else mask[:, 1:]
            d_orig, lost = d_orig[m], lost[m]
        total = d_orig.sum()
        losses.append(lost.sum() / total if total > 0 else 0.0)
    return round(100 * float(min(losses)), 1)
