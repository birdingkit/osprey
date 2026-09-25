"""Find the main animal in a photo: OWLv2 text-prompted box, then a SAM pixel mask."""

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor, SamModel, SamProcessor

DETECTOR = "google/owlv2-base-patch16-ensemble"
SEGMENTER = "facebook/sam-vit-base"
# COCO detectors have no dolphin or whale class and call flying birds kites or airplanes,
# so animals are named in text instead.
ANIMALS = ("bird", "dolphin", "whale")
# Missed birds in the 2026-08-22 set scored 0.32-0.65; an empty-sea frame scored 0.24.
MIN_SCORE = 0.3
DETECTOR_SIDE = 960  # OWLv2 input size; shrinking first avoids its slow full-res resize
SEGMENTER_SIDE = 1024  # SAM input size
CROP_PAD = 0.25  # context around the box given to SAM, as a share of the box's long side


@dataclass(frozen=True)
class Animal:
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in full-resolution pixels
    mask: np.ndarray  # bool, box-sized


class AnimalDetector:
    def __init__(self, device: torch.device):
        self.device = device
        self.detector_processor = Owlv2Processor.from_pretrained(DETECTOR)
        self.detector = Owlv2ForObjectDetection.from_pretrained(DETECTOR).eval().to(device)
        queries = [[f"a photo of a {animal}" for animal in ANIMALS]]
        self.text_inputs = self.detector_processor(text=queries, return_tensors="pt").to(device)
        self.segmenter_processor = SamProcessor.from_pretrained(SEGMENTER)
        self.segmenter = SamModel.from_pretrained(SEGMENTER).eval().to(device)

    def shrink(self, image: Image.Image) -> torch.Tensor:
        """Detector input on the CPU; thread-safe, so the photo loader runs it."""
        scale = min(1.0, DETECTOR_SIDE / max(image.size))
        small = image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.BILINEAR, reducing_gap=2.0
        )
        return self.detector_processor(images=small, return_tensors="pt")["pixel_values"]

    @torch.inference_mode()
    def __call__(self, image: Image.Image, pixels: torch.Tensor) -> Animal | None:
        """Largest animal in `image` (`pixels` is its `shrink`), or None."""
        box = self._largest_box(pixels, image.size)
        return Animal(box, self._mask(image, box)) if box else None

    def _largest_box(self, pixels: torch.Tensor, size: tuple[int, int]) -> tuple[int, int, int, int] | None:
        out = self.detector(pixel_values=pixels.to(self.device), **self.text_inputs)
        # OWLv2 pads the photo to a square, so boxes scale by the long side
        found = self.detector_processor.post_process_grounded_object_detection(
            out, threshold=MIN_SCORE, target_sizes=[(max(size), max(size))]
        )[0]
        width, height = size
        boxes = []
        for x0, y0, x1, y1 in found["boxes"].tolist():
            box = (max(0, round(x0)), max(0, round(y0)), min(width, round(x1)), min(height, round(y1)))
            if box[2] - box[0] >= 2 and box[3] - box[1] >= 2:
                boxes.append(box)
        return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), default=None)

    def _mask(self, image: Image.Image, box: tuple[int, int, int, int]) -> np.ndarray:
        """SAM mask of the animal in `box`, box-sized, judged on a padded crop so it stays detailed."""
        x0, y0, x1, y1 = box
        pad = round(max(x1 - x0, y1 - y0) * CROP_PAD)
        cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
        crop = image.crop((cx0, cy0, min(image.width, x1 + pad), min(image.height, y1 + pad)))
        # Shrink first: SamProcessor resizes full-res crops slowly on the CPU
        scale = min(1.0, SEGMENTER_SIDE / max(crop.size))
        small = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.BILINEAR)
        prompt = [(x0 - cx0) * scale, (y0 - cy0) * scale, (x1 - cx0) * scale, (y1 - cy0) * scale]
        inputs = self.segmenter_processor(small, input_boxes=[[prompt]], return_tensors="pt")
        out = self.segmenter(
            pixel_values=inputs["pixel_values"].to(self.device),
            input_boxes=inputs["input_boxes"].float().to(self.device),  # MPS has no float64
            multimask_output=False,
        )
        crop_mask = self.segmenter_processor.post_process_masks(
            out.pred_masks.cpu(), [(crop.height, crop.width)], inputs["reshaped_input_sizes"]
        )[0][0, 0].numpy()
        return crop_mask[y0 - cy0 : y1 - cy0, x0 - cx0 : x1 - cx0]
