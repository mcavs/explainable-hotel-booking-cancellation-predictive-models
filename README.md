# Explainable Hotel Booking Cancellation Prediction Models: Towards Trustworthy Survival Analysis in Hospitality

[![R](https://img.shields.io/badge/R-%E2%89%A54.3.0-blue.svg)](https://www.r-project.org/)
[![mlr3proba](https://img.shields.io/badge/Ecosystem-mlr3proba-green.svg)](https://mlr3proba.mlr-org.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation, experimental pipeline, and reproduction scripts for the study on explainable time-to-event predictive models for hotel booking cancellations and algorithmic capacity governance.

---

## Overview

Traditional cancellation prediction frameworks in hospitality treat the problem as a static binary classification task, neglecting:
1. **Temporal progression & timing:** *When* an attrition event is most likely to occur along the reservation lifecycle.
2. **Right-censoring bias:** Active bookings with future check-in dates are systematically mislabeled as non-canceled by binary classifiers.

This repository implements an end-to-end **survival analysis (time-to-event)** framework coupled with **time-dependent explainable AI (XAI)** tools (**SurvPFI** and **SurvPDP**) evaluated on a large-scale empirical cohort:
- **Empirical Cohort (Antonio et al., 2019):** Real-world Resort & City Hotel booking transactions ($N = 119,390$, 31 covariates) with verified timestamps for booking creation, arrival, and status updates, avoiding heuristic midpoint imputation.

---

## Repository Architecture

```text
├── JDSA-submission/                     # Formatted manuscript for submission
│   ├── sn-article-JDSA.tex              # Main Springer Nature LaTeX source (native TikZ)
│   ├── sn-article-JDSA.pdf              # Compiled 21-page publication PDF
│   └── sn-bibliography.bib              # BibTeX literature database
├── latex/                               # Mirror LaTeX source, styles, and figures
│   ├── sn-article.tex                   # Springer Nature manuscript
│   ├── sn-article.pdf                   # Compiled PDF
│   ├── fig_calibration_curves.png       # Figure 2: Probability calibration curves across time horizons
│   ├── fig_survpfi_ds1.png              # Figure 3: Time-dependent SurvPFI (Oblique RSF)
│   └── fig_survpdp_ds1.png              # Figure 4: Marginal SurvPDP profiles (Oblique RSF)
├── scripts/                             # Reproducible R experimental pipeline
│   ├── 00_install_packages.R            # Automated dependency installation
│   ├── 01_mlr3_experiment.R             # Data preprocessing, task creation, and 10-model training
│   ├── 02_eval_explain.R                # Evaluation metrics calculation and explainer generation
│   ├── 03_explainers_fixed.R            # Robust survival explainer wrappers
│   ├── 05_survpfi_only.R                # Continuous SurvPFI computation across time horizons
│   ├── bootstrap_evaluation.R           # 100-replicate non-parametric bootstrap resampling
│   └── hypothesis_testing.R             # Omnibus Friedman test and pairwise Wilcoxon signed-rank tests
├── results/                             # Benchmark evaluation outputs and explanations
│   ├── performance_ds1.csv              # Out-of-sample benchmark metrics across 10 survival models
│   ├── survpdp_values_vanilla_aorsf_ds1.csv   # SurvPDP marginal profile values (Oblique RSF)
│   ├── pfi_values_*.csv                 # Time-dependent Permutation Feature Importance tables
│   └── ds1_cindex_boot_matrix.rds       # Bootstrap iteration performance matrix
└── README.md
```

---

## Environment Setup & Requirements

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

## How Results are Obtained (Reproduction Guide)

Follow the sequential steps below to reproduce the empirical findings, model evaluations, and XAI plots presented in the paper.

### Step 1: Preprocessing & Time-to-Event Construction
Survival target pairs `(Y, delta)` are constructed using authentic reservation timestamps without synthetic midpoint imputation:
* For canceled reservations (`delta = 1`), `Y` is the exact elapsed duration in days between booking creation (`Arrival Date - Lead Time`) and formal status cancellation recorded in the hotel property management system.
* For checked-out reservations (`delta = 0`), `Y` is the lead time, representing right-censored observation at check-in.

Run the model training orchestration:
```bash
Rscript scripts/01_mlr3_experiment.R
```
* **Internal Resampling:** The empirical dataset is partitioned into an **80% training set** and an independent **20% held-out test set**. Stratified 10-fold cross-validation is performed within the training cohort across all 10 benchmarked survival algorithms (Cox Proportional Hazards, Elastic Net Cox, Gradient Boosted Cox, Component-wise Gradient Boosted Cox, Survival Tree, Conditional Inference Tree, Random Survival Forests, Oblique Random Survival Forests, Conditional Survival Forests, and Kaplan-Meier baseline).

### Step 2: Out-of-Sample Performance Evaluation & Statistical Testing
To evaluate predictive performance on the held-out test partition, execute:
```bash
Rscript scripts/02_eval_explain.R
Rscript scripts/bootstrap_evaluation.R
Rscript scripts/hypothesis_testing.R
```
The scripts compute out-of-sample metrics along with rigorous uncertainty measures:
1. **Harrell's Concordance Index (C-index):** Discriminative risk ranking capability.
2. **Integrated Brier Score (IBS):** Continuous time probability calibration error.
3. **Right-Censored Log-Loss (RCLL):** Probabilistic likelihood under the survival distribution.
4. **Statistical Uncertainty:** 100-replicate non-parametric bootstrap standard deviations and 95% confidence intervals, an omnibus Friedman test, and pairwise Wilcoxon signed-rank tests.

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

### Step 4: Probability Calibration Analysis
To evaluate whether models provide genuine probability estimates rather than mere relative rankings, empirical reliability diagrams across forward-looking horizons (30, 60, 90, and 180 days) are evaluated against Kaplan-Meier observed rates.

---

## Key Findings Summary

1. **The Concordance vs. Calibration Trade-off:**
   * Standard tree ensembles (such as Standard RSF) achieve high discriminative concordance (C-index > 0.85), but exhibit substantial probability calibration error.
   * **Oblique Random Survival Forests (`aorsf`)** achieve superior probability calibration (lowest Integrated Brier Score and RCLL loss), making it the most dependable algorithm for responsible capacity governance and calibrated overbooking safeguards.
2. **Compounding Price Sensitivity:**
   * SurvPDP profiles reveal that high average daily rate (ADR) reservations experience a sharply accelerating cancellation hazard as arrival approaches, reflecting strategic price-monitoring behavior.
3. **Detection of Structural Policy Anomalies:**
   * Continuous XAI successfully unmasked an empirical anomaly where non-refundable reservations exhibited elevated cancellation hazards due to speculative block bookings by intermediaries.

---

## Manuscript Compilation

The complete academic paper is formatted using Springer Nature's `sn-jnl` class with native **TikZ** vector diagrams. To compile the PDF locally:

```bash
cd latex
pdflatex -interaction=nonstopmode sn-article.tex
bibtex sn-article
pdflatex -interaction=nonstopmode sn-article.tex
pdflatex -interaction=nonstopmode sn-article.tex
```
