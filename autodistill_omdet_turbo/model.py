import os
import subprocess
from dataclasses import dataclass
from typing import Any

import torch
import numpy as np
import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel
from autodistill.helpers import load_image

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class OmdetTurbo(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology):
        try:
            from transformers import AutoProcessor, OmDetTurboForObjectDetection
        except:
            subprocess.run(
                ["pip3", "install", "git+https://github.com/huggingface/transformers"]
            )
            from transformers import AutoProcessor, OmDetTurboForObjectDetection
        self.processor = AutoProcessor.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")
        self.model = OmDetTurboForObjectDetection.from_pretrained(
            "omlab/omdet-turbo-swin-tiny-hf"
        )
        self.ontology = ontology

    def predict(self, input: Any, confidence: int = 0.1) -> sv.Detections:
        image = load_image(input, return_format="PIL")
        texts = self.ontology.prompts()
        class_idx_mapper = {label: idx for idx, label in enumerate(texts)}
        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            classes=texts,
            target_sizes=[image.size[::-1]],
            score_threshold=0.1,
            nms_threshold=0.3,
        )
        i = 0

        boxes, scores, labels = (
            results[i]["boxes"],
            results[i]["scores"],
            results[i]["classes"],
        )

        final_boxes, final_scores, final_labels = [], [], []

        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]

            if score < confidence:
                continue

            final_boxes.append(box)
            final_scores.append(score.item())
            final_labels.append(class_idx_mapper[label])

        if len(final_boxes) == 0:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array(final_boxes),
            class_id=np.array(final_labels),
            confidence=np.array(final_scores),
        )
