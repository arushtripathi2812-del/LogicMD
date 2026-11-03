"""Run the supplied LogicMD feature model on deterministic synthetic inputs.

This is only a forward-pass smoke check. The CSV is illustrative metadata; its
labels are not used to train or evaluate the model.
"""

import csv
import hashlib
from pathlib import Path

import torch

from model import DetectionModule, SimilarityModule


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "synthetic_demo.csv"


def seeded_vector(value: str, length: int) -> torch.Tensor:
    seed = int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(length, generator=generator)


def main() -> None:
    with CSV_PATH.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    text = torch.stack(
        [seeded_vector(row["text"], 200 * 200).reshape(200, 200) for row in rows]
    )
    image = torch.stack([seeded_vector(row["image"], 512) for row in rows])

    torch.manual_seed(7)
    similarity = SimilarityModule().eval()
    detector = DetectionModule().eval()
    with torch.inference_mode():
        text_aligned, image_aligned, similarity_logits = similarity(text, image)
        detection_logits = detector(text, image, text_aligned, image_aligned)

    print("LogicMD synthetic forward-pass smoke check: OK")
    print(f"Dataset rows: {len(rows)} (synthetic; labels not used)")
    print(f"Text features: {tuple(text.shape)} | image features: {tuple(image.shape)}")
    print(f"Similarity logits: {tuple(similarity_logits.shape)}")
    print(f"Detection logits: {tuple(detection_logits.shape)}")
    print("No training, dataset evaluation, or paper result is claimed.")


if __name__ == "__main__":
    main()
