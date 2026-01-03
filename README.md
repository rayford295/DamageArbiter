# 🌪️ A CLIP-Enhanced Multimodal Arbitration Framework for Explainable Disaster Damage Assessment from Street-View Imagery

This repository contains the implementation, figures, and dataset links for the paper:  
**“A CLIP-Enhanced Multimodal Arbitration Framework for Explainable Hurricane-Induced Damage Assessment from Street-View Imagery.”**

---

## 📘 Overview

This study proposes a **CLIP-enhanced multimodal arbitration framework** designed to improve the interpretability, reliability, and accuracy of street-view-based disaster damage assessment.  
It systematically combines **Vision Transformer (ViT)** and **CLIP (Contrastive Language–Image Pretraining)** representations, supported by **LLM-generated disaster annotations**.

---

## 🧩 Methodology Framework

<p align="center">
  <img src="figure/figure3.Methodology framework.png" alt="Methodology Framework" width="700">
</p>

The framework integrates:
- Vision-based feature extraction (ViT)
- LLM-assisted textual annotation generation
- CLIP-based cross-modal alignment
- Confidence-based arbitration for explainable disaster damage prediction

---

## 📊 Figures

### **Figure 1. Study Area**
<p align="center">
  <img src="figure/figure1. studyarea map.png" alt="Study Area Map" width="600">
</p>

### **Figure 2. Label Example**
<p align="center">
  <img src="figure/figure2.Label-example.png" alt="Label Example" width="600">
</p>

### **Figure 4. Vision Transformer Architecture**
<p align="center">
  <img src="figure/figure4. VIT.png" alt="ViT Architecture" width="600">
</p>

### **Figure 5. CLIP Model**
<p align="center">
  <img src="figure/figure5. clip.png" alt="CLIP Model" width="600">
</p>

### **Figure 6. Framework of Confidence-Based Error Analysis**
<p align="center">
  <img src="figure/figure6. Framework of Confidence-Based Error Analysis.png" alt="Confidence-Based Error Analysis" width="600">
</p>

### **Figure 7. Semantic Detection Pipeline**
<p align="center">
  <img src="figure/figure7. Semantic Detection.png" alt="Semantic Detection" width="600">
</p>

### **Figure 8. Spatial Mapping Results**
<p align="center">
  <img src="figure/figure8.mapping.png" alt="Mapping Results" width="600">
</p>

---

## 📂 Dataset

You can access the **street-view disaster dataset** from the following DOI:

> **Yang, Yifan (2025)**  
> *Perceiving Multidimensional Disaster Damages from Street–View Images Using Visual–Language Models*  
> [📁 figshare Dataset DOI: 10.6084/m9.figshare.28801208.v2](https://doi.org/10.6084/m9.figshare.28801208.v2)

The dataset includes:
- Pre- and post-disaster street-view imagery  
- Georeferenced location and damage type annotations  
- Severity levels (*mild*, *moderate*, *severe*)  
- Sample image regions from **Horseshoe Beach, Florida**, after **Hurricane Milton**

---
---

## 🧭 Repository Structure

📦 **CLIP-Enhanced-4hurricane**  
│  
├── 📁 **code/** — Source code for model training and evaluation  
│   ├── 🧠 `inference.py` — Inference and prediction pipeline  
│   ├── ⚙️ `train_clip.py` — CLIP model fine-tuning and multimodal arbitration  
│   └── 🧩 `utils/` — Utility functions and helper scripts  
│  
├── 📁 **dataset/** — Dataset structure and metadata  
│   ├── 🗂️ `metadata.csv` — Metadata and label information  
│   └── 🌍 `samples/` — Sample image pairs and annotations  
│  
├── 🖼️ **figure/** — Figures used in the paper and documentation  
│   ├── `figure1.studyarea map.png`  
│   ├── `figure2.Label-example.png`  
│   ├── `figure3.Methodology framework.png`  
│   ├── `figure4.VIT.png`  
│   ├── `figure5.clip.png`  
│   ├── `figure6.Framework of Confidence-Based Error Analysis.png`  
│   ├── `figure7.Semantic Detection.png`  
│   └── `figure8.mapping.png`  
│  
├── 📜 `LICENSE` — Academic research-only license  
├── 🪶 `README.md` — Project documentation  
└── 🧾 `requirements.txt` — Dependencies and environment setup  

---

## ⚠️ Usage and Permissions

All **codes, figures, and datasets** in this repository were developed and curated **solely for academic research purposes** as part of  
*“A CLIP-Enhanced Multimodal Arbitration Framework for Explainable Hurricane-Induced Damage Assessment from Street-View Imagery.”*

If you wish to **reuse, reproduce, modify, or distribute** any portion of the **codebase, figures, or dataset**, please **contact the author in advance** to obtain written permission.

📩 **Contact:**  
**Yifan Yang** ([yyang295@tamu.edu](mailto:yyang295@tamu.edu))  
Department of Geography, Texas A&M University  
🌐 [https://rayford295.github.io](https://rayford295.github.io)

🚫 Unauthorized redistribution, adaptation, or commercial use of the materials in this repository is **strictly prohibited**.


