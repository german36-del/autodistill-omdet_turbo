<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill OmdetTurbo Module

This repository contains the code supporting the OmdetTurbo base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[OmdetTurbo](https://github.com/om-ai-lab/OmDet.git), developed by Binjiang Institute of Zhejiang University, is a computer vision model for zero-shot detection in real time.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [Omdet-turbo Autodistill documentation](#TODO).

## Installation

To use OmdetTurbo with autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-omdet_turbo
```

## Quickstart

```python
from autodistill_omdet_turbo import OmdetTurbo
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import cv2

# define an ontology to map class names to our OmdetTurbo prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = OmdetTurbo(
    ontology=CaptionOntology({"person": "person", "a forklift": "forklift"})
)

results = base_model.predict("iamge.png")

plot(
    image=cv2.imread("image.png"),
    classes=base_model.ontology.classes(),
    detections=results,
)
base_model.label("./context_images", extension=".jpeg")
```


## License

This model is licensed under an [Apache 2.0](LICENSE) ([see original model implementation license](https://github.com/om-ai-lab/OmDet/tree/main?tab=Apache-2.0-1-ov-file), and the corresponding [HuggingFace Transformers documentation](https://huggingface.co/docs/transformers/main/en/model_doc/omdet-turbo)).
## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!