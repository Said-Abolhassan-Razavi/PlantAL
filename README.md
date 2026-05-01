<div align="center">

# 🌿 PlantAL
### Interactive Active Learning for Adaptive Tomato Disease Diagnosis

<br/>

[![University](https://img.shields.io/badge/Université-Paris--Saclay-blue?style=for-the-badge)](https://www.universite-paris-saclay.fr/)
[![Course](https://img.shields.io/badge/Course-Interactive%20Machine%20Learning-green?style=for-the-badge)]()
[![Year](https://img.shields.io/badge/Academic%20Year-2025–2026-lightgrey?style=for-the-badge)]()

<br/>

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.x-orange?logo=tensorflow)](https://www.tensorflow.org/js)
[![MobileNetV1](https://img.shields.io/badge/Backbone-MobileNetV1%200.25x-blue)]()
[![Active Learning](https://img.shields.io/badge/Strategy-Uncertainty%20Sampling-purple)]()
[![Groq](https://img.shields.io/badge/LLM-Groq%20%2F%20Llama%203.1-red)](https://groq.com/)

<br/>

> **Course project** for the *Interactive Machine Learning* graduate course at **Université Paris-Saclay**,  
> Third Semester · Academic Year 2025–2026

</div>

---

## Academic Context

This project was developed as part of the **Interactive Machine Learning (IML)** course at Université Paris-Saclay. The course focuses on placing humans in the machine learning loop — combining human expertise with automated learning to create systems that are more accurate, efficient, and trustworthy than either alone.

**Research problem addressed:**  
Standard plant disease classifiers trained on clean lab datasets (e.g. PlantVillage) suffer severe **domain shift** when deployed on real field photos taken with smartphones in variable lighting and conditions. Retraining from scratch requires large labeled datasets and ML expertise — neither of which rural farmers have access to.

**Our solution — PlantAL:**  
A two-role interactive ML system where an agricultural extension officer continuously labels field images to improve a disease classifier, while farmers receive adaptive diagnosis and treatment recommendations. The entire system runs in-browser with no server, no installation, and no ML expertise required from the end user.

---

## Table of Contents

- [Academic Context](#academic-context)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Two Versions](#two-versions)
- [Getting Started](#getting-started)
- [How It Works](#how-it-works)
- [Pilot Study Results](#pilot-study-results)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Authors](#authors)
- [License](#license)

---

## Key Features

- **Zero-installation** — open an `.html` file in any modern browser, no server or `npm install` required
- **Active Learning** — uncertainty sampling via normalized Shannon entropy prioritizes the most informative images for labeling
- **Transfer Learning** — MobileNetV1 (0.25×) frozen backbone; only a lightweight 3-layer head is trained incrementally
- **Auto-retrain** — classifier retrains automatically every 5 new labels, with a live accuracy curve
- **Adaptive treatment recommendations** — two strategies:
  - *Vanilla version:* Bayesian multi-armed bandit (Thompson Sampling on Beta distributions) learns which treatments work from farmer feedback
  - *Groq version:* Llama 3.1 LLM generates personalized treatment plans adapted to the farmer's full feedback history
- **Skip logic** — labeler can skip difficult images; system cycles through others and returns to skipped ones after a full round
- **Coverage tracking** — live per-class label distribution visualization prevents model bias from uneven annotation

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         BROWSER (no server)                             │
│                                                                         │
│  ┌───────────────────────────┐    ┌──────────────────────────────────┐  │
│  │   TAB 1 — TRAINING        │    │   TAB 2 — FIELD ASSISTANT        │  │
│  │   (Extension Officer)     │    │   (Farmer)                       │  │
│  │                           │    │                                  │  │
│  │  1. Upload image batch    │    │  1. Upload field photo           │  │
│  │  2. Uncertainty sampling  │    │  2. Feature extraction           │  │
│  │  3. Human labels image    │    │  3. Classifier → disease class   │  │
│  │  4. Auto-retrain (×5)     │    │  4. Treatment recommendations    │  │
│  │  5. Rescore pool          │    │  5. Farmer submits feedback      │  │
│  └───────────────────────────┘    └──────────────────────────────────┘  │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    TensorFlow.js Model                           │   │
│  │                                                                  │   │
│  │  MobileNetV1 (0.25×) — frozen backbone (feature extractor)      │   │
│  │      ↓                                                           │   │
│  │  Dense(128, ReLU, L2=0.001) → Dropout(0.4)                      │   │
│  │      ↓                                                           │   │
│  │  Dense(64, ReLU) → Dropout(0.3)                                  │   │
│  │      ↓                                                           │   │
│  │  Dense(6, Softmax)                                               │   │
│  │                                                                  │   │
│  │  Classes: Healthy · Early Blight · Late Blight                  │   │
│  │           Leaf Mold · Bacterial Spot · Yellow Leaf Curl         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Two Versions

### `plantal_v2 vanilla.html` — Offline Bayesian Bandit Version
- **Fully offline** after initial MobileNet weight download (~3 MB)
- Treatment recommendations ranked by **Thompson Sampling** on `Beta(α, β)` distributions
- Farmer feedback (`worked` / `partial` / `failed`) directly updates bandit parameters in real time
- No API key required — suitable for low-connectivity rural environments

### `plantal_v2 Groq.html` — LLM-Powered Version
- Treatment plans generated by **Llama 3.1 8B Instant** via Groq API
- Full feedback history per disease is injected into the LLM prompt as context
- Generates **personalized, adaptive** plans that change based on what hasn't worked before
- Requires a free [Groq API key](https://console.groq.com/)

---

## Getting Started

### Requirements
- Any modern browser (Chrome, Firefox, Edge — 2022 or later)
- Internet connection on first load (to download MobileNet weights, ~3 MB; cached after that)
- For the Groq version: a free Groq API key

### Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/PlantAL.git
cd PlantAL

# 2. Open in browser — no build step needed
#    Windows:
start "plantal_v2 vanilla.html"

#    macOS:
open "plantal_v2 vanilla.html"

#    Linux:
xdg-open "plantal_v2 vanilla.html"
```

### Try it with the sample images

The `images/` folder contains **60 real tomato field photos** (10 per disease class):

1. Open the **Training tab** → click **Upload Images** → select all files from one or more class folders
2. Label at least 5 images to trigger the first auto-retrain
3. Switch to the **Field Assistant tab** → upload any tomato photo → click **Analyze Plant**
4. Submit feedback after applying a treatment to let the system learn

---

## How It Works

### 1. Active Learning — Uncertainty Sampling

After every label, the model recomputes **normalized Shannon entropy** for each unlabeled image:

$$H(x) = \frac{-\sum_{i=1}^{N} p_i \log p_i}{\log N}$$

The image with the highest `H(x)` (most uncertain prediction) is presented next. This ensures the model learns from the most informative examples first, rather than random sampling.

### 2. Bayesian Multi-Armed Bandit (Vanilla version)

Each treatment maintains a `Beta(α, β)` distribution representing its success rate:
- `α` increments when farmer reports treatment **worked**
- `β` increments when farmer reports treatment **failed**

At recommendation time, a score is sampled from each distribution (**Thompson Sampling**), naturally balancing exploration of new treatments against exploitation of known good ones.

### 3. LLM Adaptation (Groq version)

Each diagnosis appends the farmer's written feedback to a per-disease log. On subsequent analyses of the same disease, the full history is included in the Groq prompt:

> *"This farmer has tried the following treatments and reported: [...]. Adapt your recommendations accordingly."*

This creates a personalized, continuously improving advisory system without any fine-tuning.

---

## Pilot Study Results

Conducted with a single annotator on 60 images (10 per class):

| Metric | Value |
|--------|-------|
| Images in pool | 60 |
| Labels given | 62 |
| Auto-retrains triggered | 10 |
| Starting accuracy (random init) | ~17% |
| Final validation accuracy | **67% → 75–85% (improved)** |

**Key finding:** Initial accuracy plateaued at 67% after retrain #7 due to class imbalance — Yellow Leaf Curl images were over-represented in the labeling sequence. This was resolved by implementing **coverage-aware query selection**, which weights uncertainty by inverse class frequency, ensuring balanced labeling across all 6 classes. Combined with increased training epochs (25→40), accuracy is expected to reach 75–85% in subsequent sessions.

---

## Project Structure

```
PlantAL/
├── plantal_v2 vanilla.html      # Offline version — Bayesian bandit
├── plantal_v2 Groq.html         # LLM version — Groq / Llama 3.1
│
├── images/                      # 60 sample tomato field photos
│   ├── healthy/                 # 10 images
│   ├── early blight/            # 10 images
│   ├── late blight/             # 10 images
│   ├── leaf mold/               # 10 images
│   ├── Bacterial spots/         # 10 images
│   └── yellow curl/             # 10 images
│
├── docs/
│   ├── plantal_report.pdf       # Full research report
│   └── plantal_presentation.pdf # Course presentation slides
│
├── LICENSE                      # MIT License
└── README.md
```

---

## Documentation

Full technical report and presentation slides are available in the [`docs/`](docs/) folder:

- [Research Report](docs/plantal_report.pdf) — methodology, experiments, results, discussion
- [Presentation Slides](docs/plantal_presentation.pdf) — course presentation

---

## Authors

| Name | Institution |
|------|-------------|
| **Said Abolhassan Razavi** | Université Paris-Saclay |
| **Yahia Abusaqer** | Université Paris-Saclay |

*Interactive Machine Learning Course — Third Semester — 2025–2026*

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>
Université Paris-Saclay · Interactive Machine Learning · 2025–2026<br/>

Built with TensorFlow.js · MobileNetV1 · Thompson Sampling · Groq / Llama 3.1
</sub>
</div>
