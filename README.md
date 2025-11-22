# BEmXRD-Nets Framework

**Official implementation for the paper:**
> **"BEmXRD-Nets Framework for Novel Machine Learning Models to Predict Crystal Energy with Diversity Structures"**
> *Samak Boonpan, Weerachai Sarakorn, Krailikhit Latpala, Pornpimon Boriwan, Pornjuk Srepusharawoot*
> *Scientific Reports (Under Review)*

---

## Overview
**BEmXRD-Nets** is a novel machine learning framework designed to predict crystal formation energy and total energy for diverse material structures, ranging from binary ($A_k B_l$) to complex quinary ($A_k B_l C_m D_n E_p$) compositions.

The framework integrates **Fundamental Atomic Properties** with **Learned Embeddings from X-ray Diffraction (XRD) patterns** to capture both chemical and structural information effectively.

---

## Important Note to Researchers (Disclaimer)

This repository provides a **demonstration of the core logic** presented in the research paper. It is designed to be lightweight and easy to reproduce.

**Please note the following:**

1.  **Partial Codebase for Demonstration:**
    This repository contains **partial code** focusing on the main architecture of the BEmXRD-Nets framework. It is intended to demonstrate the core methodology (XRD Pipeline + Atomic Weighting + Stacking Model) clearly and concisely.

2.  **Omission of String-Based Features:**
    To facilitate ease of use and demonstration, specific categorical features requiring complex string encoding (e.g., *Space Group Symbols*, *Crystal Systems*, or *Atomic Phase descriptions*) have been **intentionally omitted** in this version. This reduction simplifies the data processing pipeline for new users while retaining the essential predictive capabilities.

3.  **Main Logic Sufficiency:**
    Despite these omissions, the provided code **sufficiently demonstrates the framework's main logic and novelty**—specifically the integration of XRD embeddings with weighted atomic features. It serves as a robust baseline for understanding the proposed method.

4.  **Customization & Extension:**
    We strongly encourage researchers and users to **customize the feature extraction pipeline** (found in `src/features.py`). You can re-integrate categorical features, add new descriptors, or fine-tune the hyperparameters to further enhance prediction accuracy for your specific datasets.

---

## Key Features
* **End-to-End Pipeline:** From raw CIF (`.csv`) data to final energy prediction.
* **XRD Autoencoder:** Compresses 180-point XRD patterns into dense latent representations.
* **Advanced Atomic Features:** Automated extraction of weighted properties, stoichiometric ratios, and statistical deviations using `pymatgen` and `mendeleev`.
* **Stacking Ensemble:** Combines KRR, SVR, LightGBM, and MLP for robust regression.
* **Visualization:** Includes t-SNE projections, Parity plots, and Feature Importance analysis.

## Project Structure
```text
BEmXRD_Nets_v2/
│
├── data/raw/             # Place your dataset (df_ABCDE.csv) here
├── src/
│   ├── features.py       # Advanced Atomic Feature Extraction
│   ├── xrd_pipeline.py   # XRD Generation, Autoencoder
│   ├── models.py         # Stacking Model Definition
│   ├── config.py         # Configuration & Hyperparameters
│   └── data_loader.py    # Data Merging & Cleaning Pipeline
├── results/              # Output plots and metrics
└── main.py               # Main execution script
