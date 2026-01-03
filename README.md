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

## ⚙️ Repository Structure

