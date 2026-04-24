# Resources

This section provides access to datasets and pretrained model weights used in the Sketch2Sim framework.

---

## Datasets

The project uses multiple datasets for different stages of the pipeline.

### 1. Circuit Dataset

https://www.kaggle.com/datasets/mohammadkawsar/circuit1k

- Contains hand-drawn circuit images
- Used for complete system testing and evaluation

---

### 2. YOLOv8 Component Detection Dataset

https://www.kaggle.com/datasets/mohammadkawsar/yolo-component-data

- Used to train YOLOv8 for circuit component detection
- Includes annotated bounding boxes for circuit components

---

### 3. YOLOv8 Text Detection Dataset

https://www.kaggle.com/datasets/mohammadkawsar/yolo-text-detection-data2

- Used to train YOLOv8 for text region detection
- Includes annotated text regions in circuit diagrams

---

### 4. CRNN Text Recognition Dataset

https://www.kaggle.com/datasets/mohammadkawsar/crnn-data2

- Used to train the CRNN text recognition model
- Contains cropped text images with corresponding labels

---

## Model Weights

Pretrained model weights are not included in this repository due to their large size.

### 1. YOLOv8 Component Detection Model

https://drive.google.com/file/d/1wh14jCrszdAJ2473altf0F8hoBBzqWwU/view?usp=sharing

### 2. YOLOv8 Text Detection Model

https://drive.google.com/file/d/1b0j_S4Dq4xT5oA0jwTRBIXBfh3oeEkvU/view?usp=sharing

### 3. CRNN Text Recognition Model

https://drive.google.com/file/d/14P-ZqArVVoEHEgBEPn_HPivUpr4n_Jme/view?usp=sharing

---

## Placement

After downloading the model weights, place them in:

```text
api/model/

Example:

api/model/
├── yolo_component.pt
├── yolo_text.pt
└── crnn_model.h5
Usage Notes
Datasets and model weights are hosted externally due to size limitations.
Download the required datasets before training.
Download the model weights before running the API or end-to-end pipeline.
Update dataset and model paths in notebooks or code if your local paths differ.
Reproducibility

To reproduce the results:

Download the required datasets from Kaggle.
Download the pretrained model weights from Google Drive.
Follow the instructions in:
training/ for model training
api/ for running the web/API system