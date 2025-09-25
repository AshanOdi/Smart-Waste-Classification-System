# 🗑️ Smart Waste Classification System

## 📌 Project Overview

This project implements a **Smart Waste Classification System** using **digital image processing and deep learning**.  
The system classifies waste items into categories such as **cardboard, glass, metal, paper, plastic, and trash**, and simulates waste sorting actions.

Dataset used: [TrashNet](https://github.com/garythung/trashnet)

---

## 🚀 Features

- 📷 Accept and process images containing waste items
- 🧠 Classify waste into categories using CNN with transfer learning
- 🔄 Count and categorize processed waste items
- 📊 Maintain logs & generate statistical reports
- 🎮 Simulate/visualize sorting actions (console messages & charts)
- (Optional) Hardware integration with actuators for real-time sorting

---

## ⚙️ Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/AshanOdi/Smart-Waste-Classification-System.git
   cd Smart-Waste-Classification-System
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Dataset Preparation

1. Download **TrashNet dataset**  
   [TrashNet on Kaggle](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
2. Split into **train / validation / test** sets:
   ```bash
   python src/preprocess.py
   ```

---

## 🏋️ Training the Model

```bash
python src/train.py --epochs 20 --batch_size 32 --lr 0.001
```

The trained model will be saved in the `models/` folder.

---

## 🔍 Running Classification

Classify a single image:

```bash
python src/classify.py --image test_samples/bottle.jpg
```

Example output:

```
[RESULT] Predicted: Plastic (92%)
[SIM] Action: Divert to Bin A
```

---

## 📑 Logs & Reports

- Every classification is saved in `logs/waste_log.csv` with:

  - Timestamp
  - Image name
  - Predicted category
  - Confidence score

- Generate statistical reports:
  ```bash
  python src/report.py
  ```
  → Outputs **bar charts & pie charts** of waste distribution.

---

## 🖥️ Simulation

- Console simulation:  
  Displays a message like
  ```
  Predicted: Paper → Actuator: Bin B
  ```
- Visualization:  
  `report.py` generates real-time graphs of waste counts.

---

## ✅ Minimum Functional Requirements (Covered)

- [x] Accept and process images
- [x] Apply image classification
- [x] Count and categorize waste items
- [x] Maintain statistical logs/reports
- [x] Simulate sorting actions

---

## 🔮 Future Enhancements

- Real-time video classification
- Conveyor belt & actuator control via Raspberry Pi/Arduino
- Deploy as a **Streamlit web app** for interactive demo
