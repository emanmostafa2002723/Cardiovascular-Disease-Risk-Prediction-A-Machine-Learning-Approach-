# 🫀 Cardiovascular Disease Risk Prediction
### A Clinically-Informed Machine Learning Pipeline

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Best%20Model-orange?logo=xgboost)
![Recall](https://img.shields.io/badge/Recall-87.6%25-green)
![AUC](https://img.shields.io/badge/AUC--ROC-0.791-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **CSAI-801 · Group 18 · Winter 2024**  
> Can a machine learning pipeline optimize CVD screening sensitivity beyond standard approaches while maintaining clinical precision? **Yes.**

---

## 📌 Overview

Cardiovascular disease kills ~17.9 million people annually — the world's #1 cause of death. Traditional risk models (Framingham, SCORE2) are linear and miss complex feature interactions. This project builds a **clinically-informed ML pipeline** on 70,000 patient records, designed around a key clinical insight:

> **Missing a diagnosis is far more costly than a false alarm.**

We optimize for **recall** (sensitivity) while enforcing a minimum **precision floor of 0.62**, using threshold tuning, feature engineering, and ensemble methods to close the gap left by accuracy-focused approaches.

---

## 🏆 Key Results

| Metric | Value |
|---|---|
| **Recall** | **87.6%** — detects 87.6% of all CVD cases |
| **Precision** | **0.625** — meets the ≥0.62 clinical floor |
| **AUC-ROC** | **0.791** |
| **F1-Score** | 0.729 |
| **F2-Score** | 0.811 |
| **Accuracy** | 67.9% |
| **Optimal Threshold** | 0.310 (vs. default 0.5) |
| **Recall Gain** | **+18.3 pp** over default threshold |

The best model was **XGBoost with clinically-optimized threshold selection**, outperforming 10 other algorithms.

---

## 📁 Repository Structure

```
cvd-risk-prediction/
│
├── 📓 Final_Graduation_project_CDV_Risk_Prediction.ipynb   # Main notebook
├── 📄 Cardiovascular_Disease_Risk_Prediction.pdf           # Research paper
│
├── data/
│   └── README.md                   # Dataset download instructions
│
├── results/
│   ├── model_comparison_table.csv  # All 11 models cross-validation results
│   └── figures/                    # ROC curves, PR curves, SHAP plots
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

**Source:** [Cardiovascular Disease Dataset — Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

- **70,000** patient records
- **11** clinical and lifestyle features
- Binary target: `cardio` (0 = no CVD, 1 = CVD present)
- ~50/50 class balance — no resampling needed

> ⚠️ The dataset is not included in this repo due to size. Download it from Kaggle and place `cardio_train.csv` in the `data/` folder.

### Features Used (after selection)

| Feature | Type | Description |
|---|---|---|
| `ap_hi` | Clinical | Systolic blood pressure |
| `age_years` | Clinical | Age in years |
| `bmi` | Engineered | Weight / (Height/100)² |
| `pulse_pressure` | Engineered | ap_hi − ap_lo (arterial stiffness) |
| `weight` | Clinical | Body weight (kg) |
| `ap_lo` | Clinical | Diastolic blood pressure |
| `cholesterol_3` | Clinical (OHE) | Well above normal cholesterol |
| `gluc_3` | Clinical (OHE) | Well above normal glucose |
| `active` | Lifestyle | Physical activity (binary) |

---

## 🔬 Methodology

### 1. Preprocessing & Outlier Management
- Systolic BP clipped to **70–250 mmHg**
- Diastolic BP clipped to **40–150 mmHg**
- Enforced systolic > diastolic constraint
- `StandardScaler` fitted on training fold only (no leakage)

### 2. Feature Engineering
- **BMI** = weight / (height/100)²
- **Pulse Pressure** = ap_hi − ap_lo (ranked above raw BP in SHAP importance)
- Cholesterol & glucose one-hot encoded → 17 total features before selection

### 3. Feature Selection
Four methods evaluated (Chi², Mutual Information, ANOVA-F, RFE) across K ∈ {5, 7, 9}:
- **Winner: RFE with K=9** → AUC ≈ 0.755

### 4. Models Evaluated (11 total)

| Category | Models |
|---|---|
| Linear | Logistic Regression |
| Distance-based | K-Nearest Neighbors |
| Kernel-based | SVM (RBF kernel) |
| Tree ensemble | Random Forest, Extra Trees |
| Gradient boosting | **XGBoost** ✅, LightGBM, CatBoost, Gradient Boosting, Stacking Ensemble, Voting Ensemble |

All evaluated via **stratified 5-fold cross-validation**.

### 5. Hyperparameter Optimization
- **Optuna** — 60 trials per algorithm
- Objective: maximize cross-validated recall subject to precision ≥ 0.62

### 6. Threshold Tuning
- 300 thresholds swept from 0.10 → 0.95
- Final threshold: **0.310** (vs. default 0.5) → +18.3 pp recall gain

---

## 🧠 Feature Importance (SHAP)

Top 6 most influential features in the final XGBoost model:

1. 🩺 Systolic blood pressure (`ap_hi`)
2. 🎂 Age in years
3. 💉 Cholesterol (high)
4. 🩺 Diastolic blood pressure (`ap_lo`)
5. ⚙️ BMI (engineered)
6. ⚙️ Pulse Pressure (engineered)

> Pulse Pressure ranked **above** raw blood pressure components, validating arterial stiffness as an independent CVD risk signal.

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/cvd-risk-prediction.git
cd cvd-risk-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `cardio_train.csv` from [Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) and place it in `data/`.

### 4. Run the notebook
```bash
jupyter notebook Final_Graduation_project_CDV_Risk_Prediction.ipynb
```

---

## 📦 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
catboost>=1.0.0
optuna>=2.10.0
shap>=0.40.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

---

## 📈 Model Comparison

| Model | AUC | Recall | Precision | F1 | Accuracy |
|---|---|---|---|---|---|
| **XGBoost Tuned** ✅ | 0.793 | **0.879** | 0.622 | 0.729 | 0.676 |
| Stacking Ensemble | 0.794 | 0.876 | 0.620 | 0.726 | 0.674 |
| CatBoost | 0.794 | 0.875 | 0.622 | 0.727 | 0.676 |
| Gradient Boosting | 0.793 | 0.877 | 0.620 | 0.726 | 0.673 |
| XGBoost (base) | 0.791 | 0.876 | 0.625 | 0.729 | 0.679 |
| Random Forest | 0.790 | 0.873 | 0.623 | 0.727 | 0.676 |
| LightGBM | 0.789 | 0.868 | 0.624 | 0.726 | 0.677 |
| Logistic Regression | 0.785 | 0.865 | 0.620 | 0.722 | 0.672 |
| SVM | 0.778 | 0.847 | 0.635 | 0.726 | 0.684 |
| KNN | 0.768 | 0.804 | 0.638 | 0.711 | 0.678 |
| Extra Trees | 0.737 | 0.799 | 0.619 | 0.698 | 0.658 |

---

## ⚠️ Limitations

- Self-reported lifestyle variables (smoking, alcohol) likely contain **under-reporting bias**
- Dataset from a **single cohort** — limits generalizability
- Absence of lab biomarkers (LDL, hsCRP) caps discriminative performance at AUC ≈ 0.875

## 🔭 Future Work

- Incorporate lab-based biomarkers (LDL, hsCRP, troponin)
- SHAP-based interpretability for individual patient predictions
- Model calibration analysis (reliability diagrams)
- Prospective multi-cohort clinical validation

---

## 📚 References

1. World Health Organization. (2021). [Cardiovascular diseases (CVDs)](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds))
2. Peng et al. — XGBH model on the same Kaggle CVD dataset, AUC = 0.81
3. Mohan, S., Thirumalai, C., & Srivastava, G. (2019). Effective heart disease prediction using hybrid machine learning. *IEEE Access*, 7, 81542–81554.
4. Upadhyayula, S. K., & Pothugunta, R. (2025). Harnessing transformer models for CVD prediction. *medRxiv*.

---

## 👥 Team

**Group 18 — CSAI-801, Winter 2024**

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Built with ❤️ to help catch cardiovascular disease before it's too late.</i>
</p>
