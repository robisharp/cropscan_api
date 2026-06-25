# 🌿 CropScan: Multi-Model Crop Disease Diagnostic Engine

CropScan is an end-to-end, local-first computer vision and machine learning diagnostic platform designed to detect plant leaf diseases. Built to bypass heavy deep learning dependencies, the system extracts high-fidelity textural, structural, and chromatic feature matrices directly from raw images, classifying them through a weighted Machine Learning Ensemble pipeline.

---

## 🚀 Key Architectural Features

* **Hand-Crafted Feature Extraction:** Utilizes Histogram of Oriented Gradients (HOG) for structure, Local Binary Patterns (LBP) for micro-textures, Gray-Level Co-occurrence Matrices (GLCM) for structural spacing, and explicit RGB/HSV color segmentations.
* **Weighted ML Ensemble:** Combines **XGBoost (50%)**, **Random Forest (30%)**, and **Support Vector Machines (20%)** to eliminate individual model variance and maximize classification reliability.
* **Explainable AI (XAI):** Implements **SHAP (SHapley Additive exPlanations)** to output feature importance arrays per inference, revealing mathematically *why* the model diagnosed a specific disease.
* **Cross-Device Hybrid Deployment:** Combines a responsive static frontend deployed globally on **GitHub Pages** with a local FastAPI engine exposed through a secure **Ngrok** tunnel.

---

## 🛠️ Tech Stack & Pipeline

### Backend
* **Core Engine:** Python 3.11+, FastAPI, Uvicorn
* **Machine Learning:** XGBoost, Scikit-Learn
* **Computer Vision & Feature Arrays:** Scikit-Image, NumPy, Pillow, OpenCV
* **Model Explainability:** SHAP

### Frontend
* **Interface:** HTML5 (WebRTC Camera Stream + Native Canvas Frame Buffering)
* **Styling:** Tailwind CSS (Cyberpunk/Terminal Matrix Aesthetic)
* **Hosting:** GitHub Pages Secure Context (`https://`)

---

## 📋 Installation & Local Setup

### 1. Clone & Prepare Environment
Clone this repository to your local machine:
```bash
git clone [https://github.com/robisharp/cropscan_api.git](https://github.com/robisharp/cropscan_api.git)
cd cropscan_api

Install all necessary computer vision and machine learning dependencies:

pip install -r cropscan_requirements.txt

2. Run the FastAPI Engine
Initialize the Uvicorn application server to build and train the ensemble models locally:

Bash
uvicorn cropscan_main:app --reload --port 8000

🌐 Public Tunneling & Mobile Device Deployment
Because mobile browsers require a secure context (https://) to activate device cameras, follow this tunneling layout to link your live GitHub Pages frontend to your local computer:

1. Expose Local Server via Ngrok
In a second terminal window, start a public internet tunnel pointing to your active FastAPI server:

Bash
.\ngrok.exe http 8000
Copy the secure forwarding link provided in your terminal output window (e.g., https://unlovely-deity-repeal.ngrok-free.dev).

2. Link the Frontend
Open index.html on your computer and update line 66 with your active live tunnel link:

JavaScript
const BACKEND_URL = "https://YOUR_SUBDOMAIN_HERE.ngrok-free.dev";
3. Push to GitHub Pages
Commit your update and push to GitHub. This triggers GitHub Pages to automatically redeploy your live site:

git add index.html
git commit -m "Deploy active public backend tunnel endpoint"
git push origin main

## 🧪 Model Diagnostic Matrix

The ensemble engine is trained to classify the following plant-health targets using strict multi-spectral thresholds and spatial edge distributions:

| Class ID | Target Plant | Diagnosis Profile | Risk Level |
| :---: | :--- | :--- | :---: |
| **Class 0** | Broad Leaf | Healthy Tissue Profile (No Anomalies) | `Safe ✅` |
| **Class 1** | Tomato | Late Blight (*Fungal Infection*) | `Danger 🔥` |
| **Class 2** | Corn / Maize | Northern Leaf Blight (*Lesion Patterns*) | `Warning ⚠️` |
| **Class 3** | Apple | Apple Scab (*Canopy Retention Scabs*) | `Danger 🔥` |
| **Class 4** | Potato | Early Blight (*Concentric Ring Spots*) | `Warning ⚠️` |

## 📈 System Architecture Diagram

```mermaid
graph TD
    %% Styling Nodes
    classDef device fill:#10b981,stroke:#047857,stroke-width:2px,color:#052e16;
    classDef web fill:#1e1b4b,stroke:#312e81,stroke-width:1px,color:#e0e7ff;
    classDef local fill:#18181b,stroke:#27272a,stroke-width:1px,color:#f4f4f5;
    classDef engine fill:#065f46,stroke:#047857,stroke-width:2px,color:#ecfdf5;

    A[📸 Device Camera / File Upload]:::device -->|Captures Leaf Data| B[🌐 GitHub Pages Frontend]:::web
    B -->|Encrypted HTTPS Context| C[🔏 Ngrok Proxy Tunnel]:::web
    C -->|Secure Payload Routing| D[⚡ FastAPI Backend Engine]:::local
    
    D --> E[🖼️ Feature Engine]:::local
    D --> F[🎨 Severity Masking]:::local
    F -->|Color/Lesion Ratio| J
    
    E --> G(HOG & LBP Matrix):::local
    E --> H(HSV & GLCM Texture):::local
    
    G --> I(Scaled Vectors):::local
    H --> I
    
    I --> J[🧠 Weighted Ensemble Engine]:::engine
    
    subgraph Machine Learning Pipeline
    J -->|50%| K(XGBoost):::engine
    J -->|30%| L(Random Forest):::engine
    J -->|20%| M(Support Vector Machines):::engine
    J --> N(SHAP Explainable AI):::engine
    end
    
    J -->|Calculates Diagnostic Report| O[📝 JSON Response Output]:::device
    O -->|Renders UI Updates| B

image screenshot : <img width="653" height="901" alt="image" src="https://github.com/user-attachments/assets/e02e130d-fa32-49a8-b9bb-c2a7ad2eade8" />

by ROBISHA R P
MBA - Business Analytics
