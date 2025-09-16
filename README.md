♻️ Smart Waste Management System
📌 Project Overview

This project is developed as part of the EC9570 – Digital Image Processing course at the University of Jaffna. The system uses image processing techniques and deep learning to classify waste items into categories (e.g., cardboard, glass, metal, paper, plastic, trash). The goal is to support smart waste management by simulating sorting actions and generating statistical reports.

✨ Features

📷 Accepts and processes waste images

🧠 Classifies waste into categories using a trained deep learning model (EfficientNet/other CNN)

📊 Generates statistical logs and reports of processed waste

🎛️ Simulates sorting actions (graphical indicators, console messages, or optional hardware actuators)

✅ Achieves at least 80% accuracy on validation/testing data

🛠️ System Requirements
Software

Python 3.8+

Libraries:

OpenCV

NumPy

Torch / TensorFlow (for classification)

Matplotlib / Seaborn (for visualization)

Streamlit (for web app interface)

Hardware (Optional)

Camera/Webcam (for live capture)

Arduino/Raspberry Pi (for actuator control)

Servo motors / LEDs (for sorting simulation)

📂 Dataset

The project uses the TrashNet Dataset
 containing images in six categories:

Cardboard

Glass

Metal

Paper

Plastic

Trash

Note: All datasets must be cited if reused.

🚀 Installation & Setup

Clone the repository

git clone https://github.com/yourusername/smart-waste-management.git
cd smart-waste-management


Create a virtual environment & install dependencies

python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
pip install -r requirements.txt


Run the Streamlit app

streamlit run app.py

📊 Project Workflow

Image Acquisition & Preprocessing

Load and resize images

Apply normalization and augmentation

Waste Classification

Model trained on TrashNet dataset

EfficientNet / CNN architecture used for classification

Statistical Analysis & Reporting

Count number of items per category

Generate logs and visual reports

Simulation of Sorting

Console messages or graphical visualization

Optional actuator control with Arduino/Raspberry Pi

👥 Contributors

[Your Name] – Image Preprocessing & Model Training

[Teammate’s Name] – Web App & Statistical Reporting

🎯 Future Enhancements

Real-time classification via webcam

Integration with IoT-enabled smart bins

Recyclability detection

Deployment on embedded edge devices

📜 License

This project is for academic purposes only (University of Jaffna).
