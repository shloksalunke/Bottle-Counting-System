<div align="center">

<h1>🤖 Real-Time Bottle Counting System</h1>

<p><strong>An end-to-end Computer Vision pipeline for automated bottle detection, tracking, and counting using YOLOv8n</strong></p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00BFFF?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/Roboflow-Dataset-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/ByteTrack-Tracking-orange?style=for-the-badge" />
</p>

<p>
  <img src="https://img.shields.io/github/stars/yourusername/bottle-counting-yolov8?style=social" />
  <img src="https://img.shields.io/github/forks/yourusername/bottle-counting-yolov8?style=social" />
</p>

</div>

---

<p align="center">
  <img src="assets/thumbnail.png" width="1000">
</p>

---

## 📌 Overview

This project demonstrates a **production-style Computer Vision pipeline** — from raw data collection to real-time inference. The system detects bottles on a conveyor belt, assigns unique tracking IDs to each object, and automatically increments a count when a bottle crosses a predefined virtual line.

Built using **YOLOv8n** (~6 MB), a lightweight nano-scale model purpose-built for edge deployment scenarios, this project serves as a strong proof-of-concept for industrial inspection and automation use cases.

---

## 🏗️ System Pipeline

```
Raw Image Collection
        ↓
Annotation via Roboflow
        ↓
Data Augmentation
        ↓
YOLOv8n Fine-Tuning
        ↓
Model Evaluation & Export
        ↓
Real-Time Video Inference (OpenCV)
        ↓
Object Tracking (ByteTrack)
        ↓
Automated Bottle Counting
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🎯 Custom Object Detection | Fine-tuned YOLOv8n on a custom bottle dataset |
| 🔁 Object Tracking | Persistent unique ID assignment using ByteTrack |
| 🔢 Automated Counting | Line-crossing logic for real-time bottle counting |
| ⚡ Lightweight Model | ~6 MB YOLOv8n — optimized for edge deployment |
| 🎬 Video Output | Annotated output video with bounding boxes and count overlay |
| 📦 Modular Codebase | Clean, well-structured and easy to extend |

---

## 🛠️ Tech Stack

| Technology | Role |
|---|---|
| Python 3.10+ | Core development language |
| YOLOv8n (Ultralytics) | Object detection model |
| OpenCV | Video capture, processing & rendering |
| Roboflow | Dataset annotation & augmentation |
| ByteTrack | Multi-object tracking algorithm |

---

## 📂 Repository Structure

```
bottle-counting-yolov8/
│
├── dataset/                  # Training dataset (images + labels)
│
├── models/
│   └── best.pt               # Fine-tuned YOLOv8n weights
│
├── videos/
│   ├── input.mp4             # Source video for inference
│   └── output.mp4            # Annotated output with detections & count
│
├── assets/
│   └── output.png            # Sample output screenshot
│
├── bottle_counter.py         # Main inference & counting script
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip package manager
- A webcam or video file for testing

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/bottle-counting-yolov8.git
cd bottle-counting-yolov8

# 2. Install dependencies
pip install -r requirements.txt
```

### Running Inference

```bash
# Run on a video file
python bottle_counter.py --source videos/input.mp4

# Run on webcam
python bottle_counter.py --source 0
```

---


## 🔬 Methodology

### 1. Data Collection
Bottle images were gathered from real-world scenarios to ensure model generalization across lighting and orientation variations.

### 2. Annotation
All images were annotated using **Roboflow**, which provides a streamlined interface for bounding box labeling and dataset versioning.

### 3. Data Augmentation
To improve model robustness, the following augmentation techniques were applied via Roboflow:

- Horizontal Flip
- Contrast Adjustment
- Grayscale Conversion
- Additional Roboflow-native augmentations

### 4. Model Training
**YOLOv8n** (Nano variant) was fine-tuned on the custom dataset. The nano model was selected for its balance of speed and accuracy, making it well-suited for edge deployment scenarios.

```
Model  : YOLOv8n
Size   : ~6 MB
Format : PyTorch (.pt)
```

### 5. Tracking & Counting
**ByteTrack** was integrated to maintain consistent object IDs across frames. A virtual counting line was defined in the frame — any tracked bottle that crosses this line triggers an automated count increment.

---

## 📊 What I Learned

Building this project from scratch provided hands-on experience with the **complete CV development lifecycle**, including:

- Dataset curation and quality control strategies
- Practical annotation workflows at scale
- Effective data augmentation for small datasets
- YOLOv8 fine-tuning and hyperparameter tuning
- Multi-object tracking with persistent ID assignment
- Real-time video inference optimization
- Performance constraints and tradeoffs on lightweight edge models

---

## 🔮 Roadmap

- [ ] TensorRT optimization for faster inference
- [ ] INT8 model quantization for reduced memory footprint
- [ ] Deployment on Raspberry Pi and NVIDIA Jetson Nano
- [ ] Multi-class object counting support
- [ ] Integration with industrial defect detection workflows
- [ ] Web dashboard for live monitoring and count visualization


