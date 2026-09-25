"""Species ID with BioCLIP 2.5, scored only against birds recorded near the photo."""

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

from .detect import Bird
from .taxa import CACHE_ROOT, Species

# BioCLIP 2.5 (ViT-H/14) over BioCLIP 2 (ViT-L/14): on test seabirds 2.5 held the right
# species at ~100% across crop sizes where 2 flipped between petrel and shearwater.
MODEL = "hf-hub:imageomics/bioclip-2.5-vith14"
CACHE_DIR = CACHE_ROOT / "text-embeddings"
OUT_OF_SEASON = math.log(0.1)  # species not recorded within a month of the photo date
TOP_K = 3
CROP_PAD = 0.15  # context kept around the bird
CROP_KEEP_SIDE = 448  # 2x the classifier input, so its own resize stays high quality
TEXT_BATCH = 8  # species per text-encoder forward (x 80 prompt templates)


@dataclass(frozen=True)
class Guess:
    species: Species
    probability: float
    in_season: bool


def context_box(bird: Bird, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Square around the bird with some context, clamped to the frame."""
    width, height = image_size
    x0, y0, x1, y1 = bird.box
    side = min(round(max(x1 - x0, y1 - y0) * (1 + 2 * CROP_PAD)), width, height)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    left = min(max(cx - side // 2, 0), width - side)
    top = min(max(cy - side // 2, 0), height - side)
    return left, top, left + side, top + side


def species_crop(image: Image.Image, bird: Bird) -> Image.Image:
    """The bird's context square, shrunk to 2x the classifier input so batches stay small."""
    crop = image.crop(context_box(bird, image.size))
    crop.thumbnail((CROP_KEEP_SIDE, CROP_KEEP_SIDE), Image.LANCZOS)
    return crop


class SpeciesClassifier:
    def __init__(self, device: torch.device):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(MODEL)
        self.tokenizer = open_clip.get_tokenizer(MODEL)
        self.model.eval().to(device)
        self.device = device
        self._text: dict[int, torch.Tensor] = {}
        self._tables: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.inference_mode()
    def embed(self, crops: list[Image.Image]) -> torch.Tensor:
        """Image embeddings, one row per crop. Batch crops: on MPS a batch of 1 is ~4x slower per image.

        Each crop is averaged with its mirror image, which steadies the scores when
        the detector box shifts by a few pixels.
        """
        images = torch.stack([self.preprocess(c) for c in crops]).to(self.device)
        emb = self.model.encode_image(torch.cat([images, images.flip(-1)]), normalize=True)
        return F.normalize(emb[: len(crops)] + emb[len(crops) :], dim=-1)

    @torch.inference_mode()
    def rank(self, embedding: torch.Tensor, candidates: list[Species], in_season: set[int]) -> list[Guess]:
        txt, prior = self._table(candidates, in_season)
        logits = self.model.logit_scale.exp() * embedding @ txt.T
        probs = torch.softmax(logits + prior, dim=0)
        top = torch.topk(probs, min(TOP_K, len(candidates)))
        return [
            Guess(candidates[i], float(p), candidates[i].taxon_id in in_season)
            for p, i in zip(top.values.tolist(), top.indices.tolist())
        ]

    def _table(self, candidates: list[Species], in_season: set[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Text embedding matrix + log prior for one candidate list, built once per list."""
        key = (tuple(s.taxon_id for s in candidates), frozenset(in_season))
        if key not in self._tables:
            missing = [s for s in candidates if s.taxon_id not in self._text]
            if missing:
                self._text.update(self._load_or_encode(missing))
            txt = torch.stack([self._text[s.taxon_id] for s in candidates])
            prior = torch.tensor(
                [0.0 if s.taxon_id in in_season else OUT_OF_SEASON for s in candidates], device=self.device
            )
            self._tables[key] = (txt, prior)
        return self._tables[key]

    def _load_or_encode(self, species: list[Species]) -> dict[int, torch.Tensor]:
        """Text embeddings per species, cached on disk keyed by the exact label text.

        Labels use the recipe of BioCLIP's own TreeOfLife embeddings: taxonomy + common
        name, averaged over OpenAI's 80 ImageNet prompt templates.
        """
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        out = {}
        todo = []
        for s in species:
            text = s.clip_text()
            path = CACHE_DIR / f"{hashlib.sha1((MODEL + text).encode()).hexdigest()}.npy"
            if path.exists():
                out[s.taxon_id] = torch.from_numpy(np.load(path)).to(self.device)
            else:
                todo.append((s.taxon_id, text, path))
        if not todo:
            return out
        print(f"Encoding {len(todo)} species names with BioCLIP (one-time, cached)…")
        with torch.inference_mode():
            weights = open_clip.build_zero_shot_classifier(
                self.model,
                self.tokenizer,
                [text for _, text, _ in todo],
                open_clip.OPENAI_IMAGENET_TEMPLATES,
                num_classes_per_batch=TEXT_BATCH,
                device=self.device,
            )
        for (taxon_id, _, path), emb in zip(todo, weights.T):
            np.save(path, emb.cpu().numpy())
            out[taxon_id] = emb
        return out
