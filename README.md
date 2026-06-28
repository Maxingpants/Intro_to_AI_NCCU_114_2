# Intro_to_AI_NCCU_114_2

# ♻️ NCCU Smart Waste Classification System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green)
![Google Colab](https://img.shields.io/badge/Platform-Google%20Colab-orange)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

An intelligent **computer vision-based waste classification system** developed using **YOLOv8** to automatically detect and classify common household waste items. This project was completed as part of the Artificial Intelligence coursework at **National Chengchi University (NCCU)**.

---

# Table of Contents

* Project Overview
* Motivation
* Features
* Dataset
* Methodology
* Project Workflow
* Repository Structure
* Installation
* Usage
* Model Evaluation
* Results
* Limitations
* Future Improvements
* Technologies Used
* References
* Authors
* License
* Acknowledgements

---

# Project Overview

Efficient waste separation is essential for improving recycling efficiency and reducing environmental pollution. However, manual waste sorting is often inaccurate and time-consuming. Recent advances in deep learning have enabled object detection models capable of recognising waste materials in real time.

This project develops a smart waste classification system using the **YOLOv8** object detection framework. A custom image dataset containing twelve categories of household waste was prepared, converted into the YOLO format, and used to train a lightweight object detection model suitable for real-time applications.

The notebook demonstrates the complete machine learning workflow, including:

* Dataset preparation
* Image preprocessing
* YOLO dataset generation
* Model training
* Performance evaluation
* Interactive deployment using Gradio

---

# Motivation

Incorrect waste disposal contributes to landfill expansion, contamination of recyclable materials, and inefficient recycling systems.

The objectives of this project are to:

* automate waste recognition using computer vision;
* reduce human error during waste sorting;
* demonstrate transfer learning with YOLOv8;
* evaluate object detection performance using industry-standard metrics; and
* Provide a practical prototype that could be extended to smart recycling systems.

---

# Features

* Custom waste image dataset
* Twelve waste categories
* Automatic YOLO dataset generation
* 80/20 train–validation split
* YOLOv8 Nano transfer learning
* Object detection evaluation
* Interactive Gradio interface
* Google Drive integration
* Fully reproducible Google Colab workflow

---

# Dataset

The dataset consists of **12 waste categories**.

| ID | Waste Category    |
| -- | ----------------- |
| 0  | Plastic Bag       |
| 1  | Plastic Bottle    |
| 2  | PET Bottle        |
| 3  | Plastic Lunch Box |
| 4  | Cardboard Box     |
| 5  | Paper             |
| 6  | Glass Bottle      |
| 7  | Aluminium Can     |
| 8  | Steel Can         |
| 9  | Battery           |
| 10 | Light Bulb        |
| 11 | Other Waste       |

The dataset is automatically reorganised into the directory structure required by **Ultralytics YOLOv8**.

Dataset split:

* **Training:** 80%
* **Validation:** 20%

---

# Methodology

The project follows a standard computer vision pipeline.

## 1. Data Preparation

Raw waste images are organised into twelve class folders.

The notebook automatically:

* reads each class directory;
* creates YOLO-compatible folders;
* copies images;
* generates labels;
* creates the required `data.yaml` configuration file.

---

## 2. Data Preprocessing

The preprocessing stage includes:

* dataset organisation;
* train–validation splitting;
* directory generation;
* YAML configuration.

Google Drive is used for persistent dataset storage throughout the training process.

---

## 3. Model Training

Transfer learning is applied using the pretrained **YOLOv8 Nano** model.

Training is performed using the custom dataset:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="data.yaml")
```

YOLOv8 Nano was selected because it offers an excellent balance between:

* inference speed;
* computational efficiency;
* detection accuracy.

---

## 4. Model Evaluation

Performance is evaluated using standard object detection metrics.

| Metric       | Description                                      |
| ------------ | ------------------------------------------------ |
| Precision    | Percentage of correct detections                 |
| Recall       | Percentage of detected objects                   |
| mAP@0.5      | Mean Average Precision (IoU = 0.5)               |
| mAP@0.5:0.95 | Average precision across multiple IoU thresholds |

These metrics allow objective comparison of detection quality.

---

## 5. Deployment

After training, the best-performing model (`best.pt`) is loaded into a **Gradio** web application.

Users can:

* upload an image;
* automatically detect waste objects;
* visualise predicted bounding boxes;
* receive classification results in real time.

---

# Project Workflow

```text
Raw Images
      │
      ▼
Dataset Preparation
      │
      ▼
YOLO Dataset Formatting
      │
      ▼
Train / Validation Split
      │
      ▼
Generate data.yaml
      │
      ▼
Train YOLOv8 Nano
      │
      ▼
Model Evaluation
      │
      ▼
Inference
      │
      ▼
Gradio Interface
```

---

# Repository Structure

```text
NCCU_Smart_Waste_Classification_System/

│
├── NCCU_Smart_Waste_Classification_System.ipynb
├── README.md
├── data.yaml
│
├── images/
│   ├── train/
│   └── val/
│
├── labels/
│   ├── train/
│   └── val/
│
├── runs/
│   └── detect/
│
├── weights/
│   └── best.pt
│
└── docs/
    └── images/
```

---

# Installation

Clone the repository.

```bash
git clone https://github.com/<username>/NCCU_Smart_Waste_Classification_System.git
```

Install the required dependencies.

```bash
pip install ultralytics
pip install opencv-python
pip install gradio
pip install pyyaml
```

Alternatively,

```bash
pip install -r requirements.txt
```

---

# Usage

Open the notebook using Google Colab.

Run the notebook sequentially:

1. Mount Google Drive.
2. Prepare the dataset.
3. Generate `data.yaml`.
4. Train YOLOv8.
5. Evaluate the model.
6. Launch the Gradio interface.
7. Upload new images for prediction.

---

# Results

The notebook evaluates the trained model using:

* Precision
* Recall
* mAP@0.5
* mAP@0.5:0.95

The following figures should be included after training:

* Training loss curves
* Precision–Recall curve
* Confusion Matrix
* Validation predictions
* Sample inference images

Example repository layout:

```text
docs/images/

training_results.png
confusion_matrix.png
gradio_demo.png
prediction_example.png
```

---

# Limitations

Although the system demonstrates effective object detection, several limitations remain.

* The dataset is relatively small.
* Some waste categories contain fewer samples than others.
* Images were collected under limited lighting conditions.
* Occluded or overlapping objects may reduce detection accuracy.
* The current implementation focuses on image inference rather than live video streams.

These limitations present opportunities for future improvement.

---

# Future Improvements

Future work may include:

* increasing dataset size;
* collecting images under diverse environmental conditions;
* applying advanced data augmentation;
* comparing YOLOv8 against Faster R-CNN and EfficientDet;
* hyperparameter optimisation;
* deployment on embedded devices (Jetson Nano, Raspberry Pi);
* integration with smart recycling bins;
* mobile application deployment.

---

# Technologies Used

| Technology   | Purpose                 |
| ------------ | ----------------------- |
| Python       | Programming language    |
| Google Colab | Development environment |
| YOLOv8       | Object detection        |
| Ultralytics  | Deep learning framework |
| OpenCV       | Image processing        |
| Gradio       | User interface          |
| PyYAML       | Dataset configuration   |
| Google Drive | Cloud storage           |

---

# References

Jocher, G., Chaurasia, A., & Qiu, J. (2024). *Ultralytics YOLO Documentation*. https://docs.ultralytics.com/

Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection*. Proceedings of CVPR.

Ultralytics. (2024). *YOLOv8 Documentation*. https://docs.ultralytics.com/

OpenCV Team. (2024). *OpenCV Documentation*. https://opencv.org/

Gradio Team. (2024). *Gradio Documentation*. https://www.gradio.app/

---

# Authors

**National Chengchi University**

Artificial Intelligence Course Project

Prepared by:

* *Student Name(s)*
* Department of *Your Department*
* National Chengchi University

---

# License

This repository is provided for **academic and educational purposes only**.

Commercial use is prohibited without permission from the authors.

---

# Acknowledgements

The authors would like to acknowledge:

* National Chengchi University
* Course instructors and teaching assistants
* Ultralytics
* Google Colab
* OpenCV
* Gradio
* The Python open-source community
