"""Photo quality judged on the bird itself: sharpness and exposure."""

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter1d

from .detect import Bird

MAX_SIDE = 1024  # judge at screen-viewing scale, never upscale
BLUR_WINDOW = 11
# Label cut-offs. Calibrated on in-focus Sony A7 IV frames (58-70) vs the same
# frames with Gaussian sigma=2 or 15 px motion blur (19-38).
SHARP, SOFT = 60.0, 40.0
BLOWN_LEVEL, CRUSHED_LEVEL = 250, 5  # 8-bit levels treated as clipped
# Exposure cut-offs: test birds had 0% blown and <= 6% crushed as shot; +1 EV blew
# 1.6% of a white-bellied tern, -1 EV crushed 16% of dark petrels and boobies.
MAX_BLOWN_PCT, MAX_CRUSHED_PCT = 1.0, 10.0


def bird_sharpness(image: Image.Image, bird: Bird) -> float:
    """Sharpness of the bird's own pixels."""
    crop, mask = _masked_crop(image, bird)
    return sharpness(np.asarray(crop.convert("L")), mask)


def clipped_pct(image: Image.Image, bird: Bird) -> tuple[float, float]:
    """% of bird pixels blown out (any channel >= 250) and crushed to black (all <= 5)."""
    crop, mask = _masked_crop(image, bird)
    pixels = np.asarray(crop)[mask]
    if len(pixels) == 0:
        return 0.0, 0.0
    blown = (pixels >= BLOWN_LEVEL).any(axis=1).mean()
    crushed = (pixels <= CRUSHED_LEVEL).all(axis=1).mean()
    return round(100 * float(blown), 1), round(100 * float(crushed), 1)


def _masked_crop(image: Image.Image, bird: Bird) -> tuple[Image.Image, np.ndarray]:
    """RGB crop of the bird's box plus its mask, at most MAX_SIDE across."""
    crop, mask = image.crop(bird.box), bird.mask
    scale = min(1.0, MAX_SIDE / max(crop.size))
    if scale < 1:
        size = (round(crop.width * scale), round(crop.height * scale))
        crop = crop.resize(size, Image.LANCZOS)
        mask = np.asarray(Image.fromarray(mask).resize(size, Image.NEAREST))
    return crop, mask


def quality_label(score: float) -> str:
    return "sharp" if score >= SHARP else "soft" if score >= SOFT else "blurry"


def exposure_label(blown_pct: float, crushed_pct: float) -> str:
    """`over` / `under` mean plumage detail clipped to pure white / black."""
    if blown_pct > MAX_BLOWN_PCT:
        return "over"
    if crushed_pct > MAX_CRUSHED_PCT:
        return "under"
    return "ok"


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
    return round(100 * min(losses), 1)
