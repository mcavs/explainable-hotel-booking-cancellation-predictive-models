# Explainable Time-to-Event Predictive Models for Hotel Booking Cancellations

[![R](https://img.shields.io/badge/R-%E2%89%A54.3.0-blue.svg)](https://www.r-project.org/)
[![mlr3proba](https://img.shields.io/badge/Ecosystem-mlr3proba-green.svg)](https://mlr3proba.mlr-org.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides the official implementation, experimental pipeline, and reproduction scripts for the study:

> **"Explainable Time-to-Event Predictive Models for Hotel Booking Cancellations: A Continuous Multi-Cohort Analysis"**  
> *Under review in Electronic Markets (Special Issue on Responsible and Trustworthy AI).*

---

## 📌 Overview

Traditional cancellation prediction frameworks in hospitality treat the problem as a static binary classification task, neglecting:
1. **Temporal progression & timing:** *When* an attrition event is most likely to occur along the reservation lifecycle.
2. **Right-censoring bias:** Active bookings with future check-in dates are systematically mislabeled as non-canceled by binary classifiers.

This repository implements an end-to-end **survival analysis (time-to-event)** framework coupled with **time-dependent explainable AI (XAI)** tools (**SurvPFI** and **SurvPDP**) across two structurally distinct empirical cohorts:
- **Dataset 1 (Antonio et al., 2019):** Portuguese Resort & City Hotels ($N = 119,390$, 32 attributes, strict deposit policies).
- **Dataset 2 (Raza et al., 2023):** International Hotel Cohort ($N = 36,275$, 19 attributes, dynamic ADR pricing without deposit filters).

---

## 🏗️ Repository Architecture

```text
├── latex/                               # Manuscript LaTeX source, styles, and figures
│   ├── sn-article.tex                   # Main Springer Nature manuscript (with native TikZ)
│   ├── sn-article.pdf                   # Compiled 18-page publication PDF
│   ├── sn-bibliography.bib              # BibTeX literature database
│   ├── fig_motivation.png               # Figure 1: Censoring and time-to-event formulation
│   ├── fig_survpfi_ds1.png              # Figure 4: Time-dependent SurvPFI (Dataset 1 - Oblique RSF)
│   ├── fig_survpfi_ds2.png              # Figure 5: Time-dependent SurvPFI (Dataset 2 - Cond. Forest)
│   ├── fig_survpdp_ds1.png              # Figure 6: Marginal SurvPDP profiles (Dataset 1 - Oblique RSF)
│   └── fig_survpdp_ds2.png              # Figure 7: Marginal SurvPDP profiles (Dataset 2 - Cond. Forest)
├── scripts/                             # Reproducible R experimental pipeline
│   ├── 00_install_packages.R            # Automated dependency installation
│   ├── 01_mlr3_experiment.R             # Data preprocessing, task creation, and 10-model training
│   ├── 02_eval_explain.R                # Evaluation metrics calculation and explainer generation
│   ├── 03_explainers_fixed.R            # Robust survival explainer wrappers
│   ├── 04_survpfi_survshap.R            # Joint SurvPFI and time-dependent attribution extraction
│   ├── 05_survpfi_only.R                # Continuous SurvPFI computation across time horizons
│   ├── 06_critical_difference.R         # Statistical critical difference ranking & CD diagrams
│   └── 07_local_evaluation_fixed.R      # Out-of-sample instance-level survival evaluation
├── results/                             # Benchmark evaluation outputs and explanations
│   ├── performance_all.csv              # Summary of C-index, IBS, Schmid, and RCLL metrics
│   ├── performance_ds1.csv              # Benchmark metrics for Dataset 1
│   ├── performance_ds2.csv              # Benchmark metrics for Dataset 2
│   ├── survpdp_values_vanilla_aorsf_ds1.csv   # SurvPDP marginal values (Oblique RSF)
│   ├── survpdp_values_vanilla_cforest_ds2.csv # SurvPDP marginal values (Conditional Forest)
│   ├── pfi_values_*.csv                 # Time-dependent Permutation Feature Importance tables
│   └── plots/                           # Critical difference (CD) diagrams
└── README.md
```

---

## ⚙️ Environment Setup & Requirements

All machine learning workflows are built on **R ($\ge$ 4.3.0)** using the [`mlr3`](https://mlr3.mlr-org.com/) and [`mlr3proba`](https://mlr3proba.mlr-org.com/) ecosystems.

To install all necessary packages and their system dependencies, run:

```r
source("scripts/00_install_packages.R")
```

Key required libraries:
* **Machine Learning & Survival:** `mlr3`, `mlr3proba`, `survival`, `aorsf`, `partykit`, `ranger`, `randomForestSRC`, `glmnet`, `mboost`.
* **Explainability (XAI):** `survex`, `dalex`, `ingredients`.
* **Data Manipulation & Visualization:** `data.table`, `tidyverse`, `ggplot2`, `patchwork`.

---

## 🚀 How Results are Obtained (Reproduction Guide)

Follow the sequential steps below to reproduce the empirical findings, model evaluations, and XAI plots presented in the paper.

### Step 1: Preprocessing & Time-to-Event Construction
Survival target pairs $(Y, \delta)$ are constructed based on empirical timestamps:
* **Dataset 1:** Exact elapsed duration between booking creation and status change ($\delta = 1$) or scheduled check-in ($\delta = 0$).
* **Dataset 2:** Midpoint imputation proxy ($Y = \text{Lead\_Time} / 2$ for canceled bookings $\delta = 1$, $Y = \text{Lead\_Time}$ for non-canceled bookings $\delta = 0$).

Run the model training orchestration:
```bash
Rscript scripts/01_mlr3_experiment.R
```
* **Internal Resampling:** The data is split into an **80% training partition** and an untouched **20% held-out test partition**. Stratified 10-fold cross-validation is performed on the training split to safeguard against overfitting and ensure numerical convergence across all 10 algorithms (Cox PH, Elastic Net Cox, Gradient Boosted Cox, Component-wise GB-Cox, Survival Tree, CIT, Oblique RSF, RSF, Conditional Forest, etc.).

### Step 2: Out-of-Sample Performance Evaluation
To evaluate predictive performance on the 20% test partition without data leakage, execute:
```bash
Rscript scripts/02_eval_explain.R
```
The script computes four complementary metrics:
1. **Harrell's Concordance Index ($C$-index):** Discriminative risk ranking power.
2. **Integrated Brier Score (IBS):** Probability calibration and error over time.
3. **Integrated Schmid Score:** Alternative absolute calibration metric.
4. **Right-Censored Log-Loss (RCLL):** Information-theoretic probabilistic loss.

Outputs are exported directly to `results/performance_all.csv`.

### Step 3: Extracting Time-Dependent XAI Explanations
To compute the continuous, time-dependent feature importances and marginal survival curves on the held-out test cohort:

1. **Survival Permutation Feature Importance (SurvPFI):**
   ```bash
   Rscript scripts/05_survpfi_only.R
   ```
   *Measures the increase in Brier loss across the continuous booking lifecycle when covariates are randomly permuted.*

2. **Partial Dependence Survival Profiles (SurvPDP):**
   ```bash
   Rscript scripts/03_explainers_fixed.R
   ```
   *Isolates non-linear marginal survival curves across time for critical drivers like lead time, ADR, and deposit structures.*

### Step 4: Critical Difference Analysis
To generate the statistical critical difference (CD) diagrams comparing algorithmic rankings across metrics:
```bash
Rscript scripts/06_critical_difference.R
```
Plots are stored in `results/plots/`.

---

## 📊 Key Findings Summary

1. **The Concordance vs. Calibration Trade-off:**
   * Standard tree ensembles (e.g., RSF) achieve high discriminative concordance ($C$-index $> 0.85$), but show poor probability calibration (high Brier scores).
   * **Oblique Random Survival Forests (`aorsf`)** on Dataset 1 and **Conditional Survival Forests (`cforest`)** on Dataset 2 achieve optimal probability calibration (lowest IBS and RCLL), making them the most reliable algorithms for automated revenue management and dynamic overbooking.
2. **Compounding Price Sensitivity:**
   * SurvPDP profiles reveal that high-ADR reservations experience a sharply accelerating cancellation hazard as arrival approaches, reflecting strategic guest price-monitoring and intertemporal arbitrage.
3. **Detection of Structural Policy Anomalies:**
   * Time-dependent XAI successfully unmasked an empirical anomaly in Dataset 1, where non-refundable reservations exhibited lower survival rates than refundable ones due to speculative group reservations canceled en masse.

---

## 📜 Manuscript Compilation

The complete academic paper is formatted using Springer Nature's `sn-jnl` class with native **TikZ** vector diagrams. To compile the PDF locally:

```bash
cd latex
pdflatex -interaction=nonstopmode sn-article.tex
bibtex sn-article
pdflatex -interaction=nonstopmode sn-article.tex
pdflatex -interaction=nonstopmode sn-article.tex
```

---

## 📬 Citation & Contact

If you use this codebase or methodology in your research, please cite:

```bibtex
@article{hotel_cancellation_tte_2026,
  title = {Explainable Time-to-Event Predictive Models for Hotel Booking Cancellations: A Continuous Multi-Cohort Analysis},
  year  = {2026}
}
```
