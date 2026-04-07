# DamageArbiter

**A CLIP-Enhanced Multimodal Arbitration Framework for Hurricane Damage Assessment from Street-View Imagery**

[![arXiv](https://img.shields.io/badge/arXiv-2603.14837-b31b1b.svg)](https://arxiv.org/abs/2603.14837)
[![Dataset](https://img.shields.io/badge/Dataset-Figshare-blue)](https://doi.org/10.6084/m9.figshare.28801208.v2)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/Rayford295/BiTemporal-StreetView-Damage)

---

## Overview

DamageArbiter introduces a **disagreement-driven arbitration framework** for street-view-based disaster damage assessment. The framework combines **Vision Transformer (ViT)** and **CLIP** representations with **LLM-generated textual annotations**, using confidence-based arbitration to resolve cross-modal disagreements and produce interpretable damage predictions.

<p align="center">
  <img src="figure/figure3.Methodology framework.png" width="700">
</p>

---

## Demo

<p align="center">
  <a href="https://www.youtube.com/watch?v=MCvd-wD7Fw4">
    <img src="https://img.youtube.com/vi/MCvd-wD7Fw4/hqdefault.jpg" width="75%">
  </a>
</p>

---

## Figures

| Study Area | Label Example |
|:---:|:---:|
| <img src="figure/figure1. studyarea map.png" width="320"> | <img src="figure/figure2.Label-example.png" width="320"> |

| CLIP Architecture | Spatial Mapping Results |
|:---:|:---:|
| <img src="figure/figure5. clip.png" width="320"> | <img src="figure/figure8.mapping.png" width="320"> |

---

## Dataset

Pre- and post-disaster street-view imagery collected from **Horseshoe Beach, Florida** following **Hurricane Milton**, with georeferenced annotations and damage severity labels (*mild / moderate / severe*).

- **Figshare:** [10.6084/m9.figshare.28801208.v2](https://doi.org/10.6084/m9.figshare.28801208.v2)
- **Hugging Face:** [Rayford295/BiTemporal-StreetView-Damage](https://huggingface.co/datasets/Rayford295/BiTemporal-StreetView-Damage)

---

## Recognition

Accepted at the **AAG Annual Meeting 2026** — GIS Specialty Group Student Honors Paper Competition
**2nd Place Award**
Session: Imperial B, Ballroom Level, Hilton Union Square — March 17, 2026, 4:10–5:30 PM

---

## Citation

```bibtex
@article{yang2026damagearbiter,
  title   = {DamageArbiter: A CLIP-Enhanced Multimodal Arbitration Framework
             for Hurricane Damage Assessment from Street-View Imagery},
  author  = {Yang, Yifan and Zou, Lei and Gong, Wenjing and Fu, Kai and
             Li, Zhen and Wang, Shuo and others and Tian, H.},
  journal = {arXiv preprint arXiv:2603.14837},
  year    = {2026}
}
```

---

## Contact

**Yifan Yang** — Department of Geography, Texas A&M University
[yyang295@tamu.edu](mailto:yyang295@tamu.edu) · [rayford295.github.io](https://rayford295.github.io)

> All materials in this repository are for academic research purposes only. Please contact the author before reuse or redistribution.
