# Sketch2Sim

Dataset and implementation repository for:

**“Sketch2Sim: An End-to-End Modular Framework for Hand-drawn Circuit Analysis.”**

---

## Overview

Sketch2Sim is an end-to-end modular framework for analyzing hand-drawn circuit diagrams. The system integrates object detection and text recognition models to support circuit digitization.

The framework includes:

- YOLOv8-based component detection
- YOLOv8-based text-region detection
- CRNN-based text recognition
- Flask-based API for inference
- Training notebooks for model development and ablation studies

---

## Repository Structure

```text
Sketch2Sim-dataset-code/
├── api/          # Flask API and inference interface
├── training/     # YOLO, CRNN, and end-to-end training notebooks
├── resources/    # Dataset and pretrained model links
├── README.md
└── LICENSE
Datasets

Dataset links are provided in:

resources/README.md

The repository includes links to:

Circuit dataset
YOLO component detection dataset
YOLO text detection dataset
CRNN text recognition dataset
Model Weights

Pretrained model weights are provided through external Google Drive links in:

resources/README.md

Model files are not stored directly in this repository due to size limitations.

Running the API

Install dependencies:

pip install -r api/requirements.txt

Run:

python api/run.py

Then open:

http://127.0.0.1:5000
Training

Training notebooks are located in:

training/

They include:

YOLOv8 component detection experiments
YOLOv8 text detection experiments
CRNN text recognition training
End-to-end pipeline notebook
Reproducibility

To reproduce the workflow:

Download datasets from resources/README.md
Download model weights from resources/README.md
Update local paths in notebooks and API files
Run training notebooks or launch the Flask API
License

This repository is released under the MIT License.

Contact

For questions regarding the repository or manuscript, please contact the authors through the correspondence details provided in the paper.