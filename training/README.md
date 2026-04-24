# Training Module

This section contains all training notebooks used in the Sketch2Sim framework.

---

## Structure

- `yolo/` → YOLOv8 training and ablation experiments for component and text detection  
- `crnn/` → CRNN-based text recognition training  
- `end_to_end/` → full pipeline experiments  

---

## Training Setup

- Platform: Kaggle Notebook  
- GPU: NVIDIA Tesla (T4 / P100)  
- Frameworks: PyTorch, Ultralytics YOLOv8  

---

## YOLO Training

Includes:
- Model comparison (YOLOv8n, s, m)
- Component detection
- Text region detection
- Hyperparameter tuning

---

## CRNN Training

- Used for text recognition
- Input: cropped text regions
- Loss: CTC Loss

---

## Dataset

Dataset used for training is available at:

[PUT YOUR KAGGLE LINK HERE]

---

## Notes

- Model weights are not included in this repository  
- Training performed using GPU on Kaggle  
- Notebooks are organized for reproducibility