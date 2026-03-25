# 🌪️ DamageArbiter: A CLIP-Enhanced Multimodal Arbitration Framework for Hurricane Damage Assessment from Street-View Imagery

This repository contains the implementation, figures, and dataset links for the paper:  
**“DamageArbiter: A CLIP-Enhanced Multimodal Arbitration Framework for Hurricane Damage Assessment from Street-View Imagery.”**

---

## 📘 Overview

This study proposes a **Disagreement-driven Arbitration Framework** designed to improve the interpretability, reliability, and accuracy of street-view-based disaster damage assessment.  
It systematically combines **Vision Transformer (ViT)** and **CLIP (Contrastive Language–Image Pretraining)** representations, supported by **LLM-generated disaster annotations**.

---
## 🎥 Demo Video

<p align="center">
  <a href="https://www.youtube.com/watch?v=MCvd-wD7Fw4">
    <img src="https://img.youtube.com/vi/MCvd-wD7Fw4/hqdefault.jpg" alt="DamageArbiter Demo Video" width="85%">
  </a>
</p>

<p align="center">
  Watch the demonstration of <strong>DamageArbiter</strong> on YouTube.
</p>

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

### **Figure 5. CLIP Model**
<p align="center">
  <img src="figure/figure5. clip.png" alt="CLIP Model" width="600">
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

or

The primary hosting platform is **Hugging Face Datasets**, which provides a version-controlled repository for convenient access, inspection, and integration with machine learning workflows:

🔗 https://huggingface.co/datasets/Rayford295/BiTemporal-StreetView-Damage


The dataset includes:
- Pre- and post-disaster street-view imagery  
- Georeferenced location and damage type annotations  
- Severity levels (*mild*, *moderate*, *severe*)  
- Sample image regions from **Horseshoe Beach, Florida**, after **Hurricane Milton**

---
## 🏛️ Conference Presentation (AAG 2026)

This work has been **accepted for presentation** at the **2026 Annual Meeting of the American Association of Geographers (AAG 2026)** and received the **🥈2nd Place Award in the AAG GIS Specialty Group Student Honors Paper Competition**.

- **Session:** AAG GIS Specialty Group — **Honors Competition for Student Papers**  
- **Presentation Type:** Gallery Presentation (**Student Honors Competition**)  
- **Official AAG Link:** https://aag-meetings.secure-platform.com/aag2026/gallery/rounds/149/details/90541  

**Presentation Schedule:**
- **Date:** Tuesday, March 17, 2026  
- **Time:** 4:10 PM – 5:30 PM  
- **Location:** Imperial B, Ballroom Level, Hilton Union Square

---
## 📄 Preprint (arXiv)

Yang, Y., Zou, L., Gong, W., Fu, K., Li, Z., Wang, S., ... Tian, H. (2026).  
*DamageArbiter: A CLIP-Enhanced Multimodal Arbitration Framework for Hurricane Damage Assessment from Street-View Imagery.*  
**arXiv preprint arXiv:2603.14837**  
🔗 https://arxiv.org/abs/2603.14837

## 📌 Citation

If you find this work useful, please consider citing:

```bibtex
@article{yang2026damagearbiter,
  title={DamageArbiter: A CLIP-Enhanced Multimodal Arbitration Framework for Hurricane Damage Assessment from Street-View Imagery},
  author={Yang, Yifan and Zou, Lei and Gong, Wenjing and Fu, Kai and Li, Zhen and Wang, Shuo and others and Tian, H.},
  journal={arXiv preprint arXiv:2603.14837},
  year={2026}
}

---
## ⚠️ Usage and Permissions

All **codes, figures, and datasets** in this repository were developed and curated **solely for academic research purposes** as part of  
*“ DamageArbiter: A Disagreement-driven Arbitration Framework for Hurricane Damage Assessment from Street-View Imagery.”*

If you wish to **reuse, reproduce, modify, or distribute** any portion of the **codebase, figures, or dataset**, please **contact the author in advance** to obtain written permission.

📩 **Contact:**  
**Yifan Yang** ([yyang295@tamu.edu](mailto:yyang295@tamu.edu))  
Department of Geography, Texas A&M University  
🌐 [https://rayford295.github.io](https://rayford295.github.io)

🚫 Unauthorized redistribution, adaptation, or commercial use of the materials in this repository is **strictly prohibited**.


