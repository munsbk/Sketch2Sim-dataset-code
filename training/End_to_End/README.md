# End-to-End Pipeline

This notebook demonstrates the full Sketch2Sim pipeline, integrating:

- Component detection (YOLOv8)
- Text region detection
- Text recognition (CRNN)

---

## ⚠️ Important Setup

Before running the notebook, you must update the model paths.

### 🔹 Replace Model Paths

The notebook currently references model weights using Google Drive paths.

👉 You MUST replace these paths with your own local or Drive paths.

Example:

```python
# ❌ Original (will not work for you)
yolo_model_path = "/content/drive/MyDrive/your_path/yolo_component.pt"

# ✅ Update to your path
yolo_model_path = "/your/local/path/yolo_component.pt"
📥 Required Models

Download the required model weights from:

api/model/README.md

Then update the notebook paths accordingly.

▶️ How to Run
Open the notebook in Google Colab
Upload or mount your Google Drive
Update all model paths
Run cells sequentially
🧠 Notes
This notebook assumes all trained models are available
It will not run correctly without updating model paths
Output depends on correct placement of YOLO and CRNN models