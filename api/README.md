# API Module

This module provides a Flask-based web API for the Sketch2Sim framework. It enables users to upload hand-drawn circuit images and processes them through the detection and text extraction pipeline.

---

## 📌 Overview

The API performs the following tasks:

- Accepts an input circuit image
- Runs component detection using YOLOv8
- Detects text regions
- Extracts text using CRNN with CTC loss
- Returns processed results

---

## 📂 Folder Structure

```text
api/               
├── run.py                # Main Flask application
├── model.py              # Model loading and inference logic
├── requirements.txt      # Python dependencies
├── README.md
│
├── model/                # Model directory (weights not included)
│   └── README.md
│
├── static/               # CSS, JS, and static assets
│   ├── css/
│   └── js/
│
├── templates/            # HTML templates
│
├── uploads/              # Uploaded images (runtime)
│   └── .gitkeep
│
├── outputs/              # Processed outputs (runtime)
│   └── .gitkeep
