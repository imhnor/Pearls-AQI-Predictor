# Pearls AQI Predictor

### End-to-End MLOps Pipeline for Air Quality Forecasting

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![MLOps](https://img.shields.io/badge/MLOps-Hopsworks-green.svg)](https://www.hopsworks.ai/)
[![CI/CD](https://img.shields.io/badge/CI/CD-GitHub_Actions-orange.svg)](https://github.com/features/actions)

---

## Overview

**Pearls AQI Predictor** is a production-ready MLOps project that forecasts the **Air Quality Index (AQI) of Lahore** using automated data pipelines, feature engineering, and machine learning models.

The project integrates **Hopsworks Feature Store**, **Model Registry**, and **GitHub Actions CI/CD** to create a fully automated workflow for data ingestion, feature generation, model training, versioning, and deployment-ready inference.

---

## MLOps Core

### Hopsworks Integration

#### Feature Store

* Centralized storage for engineered features.
* Automated creation of temporal lag features (`t-1`, `t-2`, `t-3`).
* Consistent feature availability across training and inference pipelines.

#### Model Registry

* Automated registration of trained models.
* Version-controlled repository for model management.
* Supports both traditional ML and deep learning models.

#### Dynamic Model Versioning

* Custom logic automatically increments model versions.
* Prevents accidental overwrites of existing models.
* Ensures full reproducibility and model traceability.

---

### CI/CD Pipeline (GitHub Actions)

#### Automated Workflows

* Hourly cron jobs trigger the feature engineering pipeline.
* Continuous data ingestion and feature updates.
* Minimal manual intervention.

#### Decoupled Training Architecture

* Training workflow runs independently of data ingestion.
* Models can be retrained on demand.
* Improved scalability and resource utilization.

---

### Training Strategy

#### Hybrid Modeling Approach

The project benchmarks multiple forecasting models:

| Category                   | Models                     |
| -------------------------- | -------------------------- |
| Classical Machine Learning | Random Forest, XGBoost     |
| Deep Learning              | GRU (Gated Recurrent Unit) |

#### Reproducibility

* Standardized preprocessing pipeline.
* Consistent feature transformations.
* Reproducible training and evaluation workflows.

---

## Project Architecture

```text
Data Source
     │
     ▼
Feature Engineering Pipeline
     │
     ▼
Hopsworks Feature Store
     │
     ├────────► Model Training
     │              │
     │              ▼
     │       Random Forest
     │       XGBoost
     │       GRU
     │              │
     ▼              ▼
Model Registry ◄────┘
     │
     ▼
Gradio Dashboard
```

---

## Repository Structure

```text
Pearls_AQI_Predictor/
├── .github/
│   └── workflows/
│       └── pipeline.yml          # CI/CD & Scheduled Workflows
│
├── src/
│   ├── processor.py             # Data Cleaning & Preprocessing
│   ├── pipeline.py              # Feature Engineering Pipeline
│   ├── train.py                 # Model Training & Registration
│   └── app.py                   # Gradio Inference Dashboard
│
├── .env                         # API Keys & Secrets
├── requirements.txt             # Project Dependencies
└── README.md                    # Documentation
```

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Pearls-AQI-Predictor.git

cd Pearls-AQI-Predictor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root directory:

```env
HOPSWORKS_API_KEY=your_key_here
OPENWEATHER_API_KEY=your_key_here
```

### 4. Run the Dashboard

```bash
python src/app.py
```

### 5. Train Models

```bash
python src/train.py
```

### 6. Automated Pipeline

The feature engineering and data ingestion pipeline runs automatically through the GitHub Actions workflow according to the configured cron schedule.

---

## Technology Stack

* Python 3.11
* Hopsworks Feature Store
* Hopsworks Model Registry
* GitHub Actions
* XGBoost
* Random Forest
* TensorFlow / Keras (GRU)
* Pandas
* NumPy
* Scikit-learn
* Gradio

---

## Future Improvements

* Multi-step AQI forecasting
* Hyperparameter optimization using Optuna
* Model monitoring and drift detection
* Real-time data streaming
* Docker containerization
* Kubernetes deployment

---
