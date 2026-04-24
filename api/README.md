# API Module

This module provides a Flask-based web API for the Sketch2Sim framework. It enables users to upload hand-drawn circuit images and process them through the detection and text extraction pipeline.

---

## Overview

The API performs the following tasks:

- Accepts an input circuit image
- Runs component detection using YOLOv8
- Detects text regions
- Extracts text using CRNN with CTC loss
- Returns processed results

---

## Folder Structure

```text
api/
├── run.py                # Main Flask application
├── model.py              # Model loading and inference logic
├── requirements.txt      # Python dependencies
├── README.md
│
├── model/                # Model directory; weights are not included
│   └── README.md
│
├── static/               # CSS, JS, fonts, and static assets
│   ├── css/
│   ├── js/
│   └── font/
│
├── templates/            # HTML templates
│
├── uploads/              # Uploaded images during runtime
│   └── .gitkeep
│
└── outputs/              # Processed outputs during runtime
    └── .gitkeep
## Installation

```bash
pip install -r requirements.txt
python run.py


Model Weights

Model weight files are not included in this repository due to file-size limitations.

Please download the required weights from the links provided in:

api/model/README.md
