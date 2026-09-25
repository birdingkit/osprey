"""Find birds in a photo: box + pixel mask per bird (COCO Mask R-CNN)."""

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.transforms.functional import to_tensor

MIN_SCORE = 0.5


@dataclass(frozen=True)
class Bird:
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in full-resolution pixels
    mask: np.ndarray  # bool, box-sized

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    def within(self, crop_box: tuple[int, int, int, int]) -> "Bird":
        """The same bird in the coordinates of an image cropped to `crop_box`."""
        dx, dy = crop_box[0], crop_box[1]
        x0, y0, x1, y1 = self.box
        return Bird((x0 - dx, y0 - dy, x1 - dx, y1 - dy), self.mask)


class BirdDetector:
    def __init__(self, device: torch.device):
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.bird_label = weights.meta["categories"].index("bird")
        self.model = maskrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=MIN_SCORE)
        self.model.eval().to(device)
        self.device = device

    def shrink(self, image: Image.Image) -> tuple[torch.Tensor, float]:
        """Model input on the CPU plus its scale; thread-safe, so the photo loader runs it.

        Shrinks to the size the model would rescale to anyway (short side 800, long
        side <= 1333), so the 33 MP frame is resampled once.
        """
        transform = self.model.transform
        scale = min(1.0, transform.min_size[0] / min(image.size), transform.max_size / max(image.size))
        small = image.resize((round(image.width * scale), round(image.height * scale)), Image.BILINEAR)
        return to_tensor(small), scale

    @torch.inference_mode()
    def __call__(self, shrunk: tuple[torch.Tensor, float]) -> list[Bird]:
        """Birds found in a `shrink`-ed photo, boxes and masks in full-resolution pixels."""
        x, scale = shrunk
        out = self.model([x.to(self.device)])[0]
        birds = []
        for box, label, mask in zip(out["boxes"], out["labels"], out["masks"]):
            if label != self.bird_label:
                continue
            x0, y0, x1, y1 = (round(v) for v in box.tolist())
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            small_mask = mask[0, y0:y1, x0:x1].cpu().numpy()
            full_box = tuple(round(v / scale) for v in (x0, y0, x1, y1))
            full_size = (full_box[2] - full_box[0], full_box[3] - full_box[1])
            full_mask = np.asarray(Image.fromarray(small_mask).resize(full_size, Image.BILINEAR)) > 0.5
            birds.append(Bird(full_box, full_mask))
        return sorted(birds, key=lambda b: b.area, reverse=True)
