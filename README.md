# 🧠 Stroke Risk Classifier

A machine learning project that builds, tunes, and compares multiple classification models to predict stroke risk based on patient health data. The project addresses a heavily imbalanced dataset (~95% no-stroke, ~5% stroke) and optimizes for **recall on the minority stroke class** — because in a medical setting, missing a real stroke is far more dangerous than a false alarm.

---

## 📊 Model Comparison

![Model Comparison](images/model_comparison.png)

The three tuned models (Logistic Regression, Decision Tree, Random Forest) dramatically outperform their defaults and both KNN variants. Default models and KNN essentially failed to detect stroke cases, with stroke recall below 0.20.

---

## 🏆 Results Summary

| Model | Recall Macro | Stroke Recall |
|---|---|---|
| ✅ Logistic Regression (tuned) | ~0.81 | ~0.88 |
| Decision Tree (tuned) | ~0.79 | ~0.88 |
| Random Forest (tuned) | ~0.77 | ~0.80 |
| Decision Tree (default) | ~0.64 | ~0.35 |
| KNN (default) | ~0.57 | ~0.18 |
| KNN (tuned) | ~0.56 | ~0.18 |
| Random Forest (default) | ~0.53 | ~0.08 |
| Logistic Reg (default) | ~0.52 | ~0.08 |

---

## 🏅 Recommended Model

**Tuned Logistic Regression** is recommended for production because:
- Highest recall macro (~0.81) and stroke class recall (~0.88)
- `class_weight='balanced'` handles the 95/5 imbalance without manual resampling
- Fully interpretable — coefficients convert to odds ratios, auditable by clinicians
- Simplest top-performing model, least likely to overfit on new patient data

---

## 🧪 Models & Hyperparameters Tuned

All models were tuned using `GridSearchCV` with `scoring='recall_macro'` and 3-fold cross-validation.

| Model | Hyperparameters Tuned |
|---|---|
| Decision Tree | `max_depth`, `min_samples_leaf`, `class_weight` |
| Logistic Regression | `solver`, `penalty`, `C`, `class_weight`, `l1_ratio` |
| K-Nearest Neighbors | `n_neighbors`, `weights`, `metric`, `p` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`, `class_weight` |

---

## 📁 Project Structure
stroke-risk-classifier/
│
├── images/
│   └── model_comparison.png
├── Decision_Tree_LogReg_Random_Forest_or_KNN.ipynb
└── README.md
---

## ⚙️ Requirements033
pandas
numpy
matplotlib
scikit-learn

---

## 📂 Dataset

[Stroke Prediction Dataset](https://drive.google.com/file/d/1LRFYUMFr8YnpqduFBHu2TMxL3ZaJVBKB/view?usp=sharing)

| Property | Value |
|---|---|
| Target | `stroke` (1 = stroke, 0 = no stroke) |
| Class balance | ~95% no-stroke, ~5% stroke |
| Missing values | `bmi` (52 missing → imputed with median) |

---

## 👤 Author

Ali Abusohiban — [GitHub](https://github.com/aliabusohiban)
