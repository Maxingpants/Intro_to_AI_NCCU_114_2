# ♻️ NCCU Smart Waste Classification System

> **Deep Learning-Based Waste Detection and Classification Using YOLOv8**

Artificial Intelligence Final Project
National Chengchi University (NCCU)

---

## Abstract

Waste sorting plays a critical role in improving recycling efficiency and reducing environmental pollution. Manual waste classification, however, is often labour-intensive, inconsistent, and susceptible to human error. Recent advances in computer vision provide an opportunity to automate this process through deep learning.

This project presents a **smart waste classification system** built using the **YOLOv8 object detection framework**. A custom dataset containing twelve categories of household waste was prepared, converted into the YOLO annotation format, and used to train a lightweight object detection model through transfer learning. The project demonstrates the complete machine learning pipeline, including dataset preparation, preprocessing, model training, evaluation, and deployment using a Gradio web interface.

The repository is designed to provide a reproducible implementation of an end-to-end computer vision workflow for intelligent waste detection.

---

# Table of Contents

* [Project Overview](#project-overview)
* [Research Objectives](#research-objectives)
* [System Overview](#system-overview)
* [Repository Structure](#repository-structure)
* [Dataset](#dataset)
* [Methodology](#methodology)
* [Experimental Results](#experimental-results)
* [Installation](#installation)
* [Running the Project](#running-the-project)
* [Technologies Used](#technologies-used)
* [Limitations](#limitations)
* [Future Work](#future-work)
* [Reproducibility](#reproducibility)
* [References](#references)
* [Authors](#authors)
* [Acknowledgements](#acknowledgements)
* [License](#license)

---

# Project Overview

Improper waste disposal remains one of the major contributors to environmental degradation. Incorrectly sorted waste contaminates recyclable materials, increases landfill usage, and reduces recycling efficiency.

The objective of this project is to investigate the application of modern object detection techniques for automated waste classification. Using **YOLOv8 Nano**, the system detects and classifies multiple categories of waste objects within an image, providing an efficient and scalable approach for intelligent recycling systems.

This project demonstrates:

* Custom dataset preparation
* Automatic YOLO dataset generation
* Transfer learning using YOLOv8
* Model evaluation using standard detection metrics
* Interactive deployment through Gradio

---

# Research Objectives

The primary objectives of this project are:

* Develop an intelligent waste classification system using deep learning.
* Construct a custom object detection dataset for household waste.
* Investigate the effectiveness of YOLOv8 for waste detection.
* Evaluate model performance using industry-standard metrics.
* Develop an interactive prototype suitable for real-time inference.
* Produce a fully reproducible machine learning workflow.

---

# System Overview

The complete workflow implemented in this repository is illustrated below.

```text
Raw Images
      │
      ▼
Dataset Preparation
      │
      ▼
YOLO Annotation Generation
      │
      ▼
Train / Validation Split
      │
      ▼
Generate data.yaml
      │
      ▼
YOLOv8 Training
      │
      ▼
Model Evaluation
      │
      ▼
Image Inference
      │
      ▼
Gradio Deployment
```

### System Architecture

> **Replace with an architecture diagram**

```
docs/images/system_architecture.png
```

---

# Repository Structure

```text
NCCU_Smart_Waste_Classification_System/

│
├── NCCU_Smart_Waste_Classification_System.ipynb
├── README.md
├── requirements.txt
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

# Dataset

The custom dataset contains **12 household waste categories**.

| ID | Category          |
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

Dataset split:

| Dataset    | Percentage |
| ---------- | ---------: |
| Training   |        80% |
| Validation |        20% |

The notebook automatically organises the dataset into the directory structure required by Ultralytics YOLOv8.

---

# Methodology

## 1. Data Collection

Images representing twelve categories of household waste were collected and organised into class-specific directories. The dataset includes recyclable and non-recyclable materials commonly encountered in Taiwan.

---

## 2. Data Preprocessing

Prior to training, the notebook automatically:

* validates image files;
* organises images into class folders;
* creates the required directory structure;
* generates YOLO annotations;
* constructs the `data.yaml` configuration file;
* performs an 80:20 train–validation split.

Google Drive is used to store both datasets and trained model weights.

---

## 3. Model Selection

This project employs **YOLOv8 Nano**, a lightweight object detection model developed by Ultralytics.

YOLOv8 Nano was selected because it offers:

* fast inference speed;
* low computational cost;
* efficient transfer learning;
* suitability for deployment on embedded devices.

Transfer learning is applied using pretrained COCO weights.

---

## 4. Model Training

Training is performed using the Ultralytics implementation of YOLOv8.

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640
)
```

---

## 5. Model Evaluation

Model performance is evaluated using the following metrics.

| Metric       | Description                                  |
| ------------ | -------------------------------------------- |
| Precision    | Detection accuracy                           |
| Recall       | Object retrieval performance                 |
| mAP@0.5      | Mean Average Precision (IoU = 0.5)           |
| mAP@0.5:0.95 | Mean Average Precision across IoU thresholds |

These metrics provide quantitative assessment of the detector's performance.

---

## 6. Deployment

Following training, the best-performing model (`best.pt`) is deployed using **Gradio**.

Users may:

* upload an image;
* detect multiple waste objects;
* visualise predicted bounding boxes;
* receive classification results in real time.

---

# Experimental Results

The trained model generates several evaluation outputs.

### Training Curves

> Replace with:

```
docs/images/results.png
```

---

### Confusion Matrix

> Replace with:

```
docs/images/confusion_matrix.png
```

---

### Precision–Recall Curve

> Replace with:

```
docs/images/pr_curve.png
```

---

### Sample Predictions

> Replace with:

```
docs/images/predictions.png
```

---

### Model Performance

Replace these values with your actual training results.

| Metric       | Score |
| ------------ | ----- |
| Precision    | XX    |
| Recall       | XX    |
| mAP@0.5      | XX    |
| mAP@0.5:0.95 | XX    |

---

# Installation

Clone the repository.

```bash
git clone https://github.com/<username>/NCCU_Smart_Waste_Classification_System.git

cd NCCU_Smart_Waste_Classification_System
```

Install the required dependencies.

```bash
pip install -r requirements.txt
```

or

```bash
pip install ultralytics
pip install opencv-python
pip install gradio
pip install pyyaml
```

---

# Running the Project

Open the notebook using **Google Colab**.

Execute the notebook sequentially.

1. Mount Google Drive.
2. Install dependencies.
3. Prepare the dataset.
4. Generate `data.yaml`.
5. Train YOLOv8.
6. Evaluate the model.
7. Launch the Gradio interface.
8. Upload images for prediction.

---

# Technologies Used

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Programming language      |
| YOLOv8       | Object detection          |
| Ultralytics  | Deep learning framework   |
| OpenCV       | Image processing          |
| Gradio       | Interactive web interface |
| Google Colab | Cloud development         |
| Google Drive | Data storage              |
| PyYAML       | Dataset configuration     |

---

# Limitations

Although the proposed system demonstrates promising performance, several limitations remain.

* Limited dataset size.
* Class imbalance across waste categories.
* Images collected under relatively controlled conditions.
* Detection performance decreases under severe occlusion.
* The current implementation focuses on still images rather than live video.

Future work should address these limitations through larger datasets and improved data augmentation.

---

# Future Work

Potential extensions include:

* expanding the dataset with additional waste categories;
* collecting images under diverse environmental conditions;
* applying advanced augmentation techniques;
* hyperparameter optimisation;
* comparing YOLOv8 with Faster R-CNN and EfficientDet;
* deployment on NVIDIA Jetson or Raspberry Pi;
* real-time webcam inference;
* integration into smart recycling bins;
* mobile application deployment.

---

# Reproducibility

The repository provides a complete, reproducible workflow.

The recommended execution order is:

1. Install dependencies.
2. Mount Google Drive.
3. Prepare the dataset.
4. Generate YOLO annotations.
5. Train the model.
6. Evaluate the model.
7. Launch the Gradio application.

The notebook was developed using:

* Python 3.11
* Google Colab
* Ultralytics YOLOv8

---

# References

Jocher, G., Chaurasia, A., & Qiu, J. (2024). *Ultralytics YOLO Documentation*. https://docs.ultralytics.com/

Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.

OpenCV Development Team. (2024). *OpenCV Documentation*. https://opencv.org/

Gradio Team. (2024). *Gradio Documentation*. https://www.gradio.app/

---

# Authors

**National Chengchi University**

Artificial Intelligence Final Project

**Team Members**

* Edward Ladham 113ZU1031
* 柯立宸 112104039
* *Student 3*

---

# Acknowledgements

The authors would like to thank:

* National Chengchi University
* Course instructors and teaching assistants
* Ultralytics
* Google Colab
* OpenCV
* Gradio
* The Python open-source community

---

# License

This repository is released for **academic and educational purposes**.

Commercial use is prohibited without permission from the project authors.

If this repository contributes to your work, please provide appropriate attribution.
