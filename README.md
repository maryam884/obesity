# Obesity Level Prediction — ML Project

Multi-class classification using supervised and unsupervised machine learning on the UCI Obesity Estimation dataset.

## Project Structure

```
OBESITY_APP/
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── kmeans.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── cm_lo.npy       ← Logistic Regression confusion matrix
│   ├── cm_rf.npy       ← Random Forest confusion matrix
│   └── cm_gb.npy       ← Gradient Boosting confusion matrix
├── app.py                          ← Streamlit dashboard
├── train_models.py                 ← Training script
├── ObesityDataSet_raw_and_data_sinthetic-2.csv
├── eda_data.csv
├── comparison.json
├── feature_names.json
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Train Models

```bash
python train_models.py
```

## Run Streamlit App

```bash
streamlit run app.py
```

## Dataset

- **Source**: UCI ML Repository — Obesity Levels Estimation
- **Samples**: 2,111
- **Features**: 16 (demographics, eating habits, physical activity)
- **Target**: 7 obesity level classes

## Models

| Model | Accuracy | AUC (macro) |
|---|---|---|
| Logistic Regression | 87.0% | 98.5% |
| Random Forest | 95.3% | 99.7% |
| Gradient Boosting | **96.0%** | **99.7%** |
| KMeans (k=7) | Silhouette: 0.152 | — |

## Key Findings

- **Weight** is the most predictive feature (35.2% RF importance)
- Gradient Boosting is the best overall model
- Obesity Type III and Insufficient Weight are the most distinct classes
- Overweight Level I/II show the most overlap — expected given their biological proximity
