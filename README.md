<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=200&section=header&text=⚡%20XGBoost%20Binary%20Classifier&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=High-Performance%20Classification%20on%201M%20Samples%20|%2097.37%25%20Accuracy&descAlignY=60&descAlign=50" width="100%"/>

---

## 📌 Project Overview

This project benchmarks **XGBoost (Extreme Gradient Boosting)** on a large-scale synthetic binary classification task with **1,000,000 samples and 30 features**. The goal is to demonstrate XGBoost's performance, speed, and scalability — achieving **97.37% accuracy** on 200K held-out test samples with a balanced class distribution.

XGBoost's ensemble of gradient-boosted decision trees makes it the go-to model for structured/tabular data, outperforming many deep learning approaches on such tasks.

---

## 📂 Dataset

| Property | Value |
|:---|:---|
| Source | `sklearn.datasets.make_classification` (synthetic) |
| Total Samples | 1,000,000 |
| Features | 30 total (15 informative, 2 redundant, 13 noise) |
| Classes | Binary (0 / 1) — balanced |
| Train Split | 800,000 samples (80%) |
| Test Split | 200,000 samples (20%) |
| Random State | 42 |

---

## 🔄 Pipeline Workflow

```
Raw Synthetic Data → Train/Test Split (80/20) → XGBClassifier Init → Model Training → Prediction → Evaluation
```

1️⃣ **Data Generation** — Synthetic dataset with 1M samples, 30 features (15 informative) using `make_classification`

2️⃣ **Train/Test Split** — Stratified 80/20 split using `train_test_split` with `random_state=42`

3️⃣ **Model Initialization** — `XGBClassifier` with 1000 estimators, `max_depth=3`, `learning_rate=0.1`

4️⃣ **Training** — Gradient boosted trees fitted on 800K training samples

5️⃣ **Evaluation** — Accuracy score + full classification report on 200K test samples

---

## 🤖 Models

### 1️⃣ XGBClassifier ⭐ Best Model

```python
model_xgbc = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
model_xgbc.fit(X_train, y_train)
```

- **Ensemble method:** Boosting — sequentially corrects errors of prior trees
- **Shallow trees** (`max_depth=3`) prevent overfitting while maintaining expressiveness
- **1000 estimators** with a low learning rate ensures slow, stable convergence
- Built on C++ backend — handles 1M samples efficiently without batching

---

## 📊 Results

| Model | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1 (0) | F1 (1) | Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🏆 **XGBClassifier** | **0.97** | **0.98** | **0.98** | **0.97** | **0.97** | **0.97** | **97.37%** |

**Detailed Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| Class 0 | 0.97 | 0.98 | 0.97 | 100,294 |
| Class 1 | 0.98 | 0.97 | 0.97 | 99,706 |
| **Macro Avg** | **0.97** | **0.97** | **0.97** | **200,000** |

---

## 🔍 Key Insights

- ⚡ **XGBoost scales to 1M samples** with no memory issues, demonstrating real production-readiness for large tabular datasets
- 🎯 **97.37% accuracy** achieved on a balanced 30-feature problem — both classes predicted with near-equal precision/recall
- 🌳 **Shallow trees (max_depth=3)** + **low learning rate (0.1)** is a classic and effective XGBoost config — avoids overfitting on high-dimensional data
- 📉 With 15 informative features out of 30, XGBoost naturally de-weights noisy/redundant features via gradient-based feature selection
- 🔁 **1000 estimators** with a slow learning rate outperforms fewer deep trees — gradient boosting rewards patience

---

## 🗂️ Repository Structure

```
xgboost-classifier/
│
├── XGBoost.ipynb          
└── README.md              

```

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/ronakrajput8882/xgboost-classifier.git
cd xgboost-classifier

# Install dependencies
pip install xgboost scikit-learn numpy

# Launch the notebook
jupyter notebook XGBoost.ipynb
```

---

## 🧠 Key Learnings

- XGBoost's regularization (via `max_depth` and `learning_rate`) is critical for preventing overfitting even on large synthetic datasets
- Gradient boosting sequentially reduces residual errors — 1000 weak learners combine into a very strong classifier
- Balanced class distributions (50/50 split) allow raw accuracy to be a valid metric — no class weighting needed
- `make_classification` is an excellent benchmarking tool for validating model pipelines before applying to real-world data
- XGBoost v3.x brings performance improvements — always pin your version in production

---

## 🛠️ Tech Stack

| Tool | Use |
|:---|:---|
| Python 3.10+ | Core language |
| XGBoost 3.2.0 | Gradient boosted classifier |
| scikit-learn | Train/test split, metrics |
| NumPy | Array operations |
| Jupyter Notebook | Development environment |

---

<div align="center">

### Connect with me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ronakrajput8882)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/techwithronak)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ronakrajput8882)

*If you found this useful, please ⭐ the repo!*

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=100&section=footer" width="100%"/>

</div>
