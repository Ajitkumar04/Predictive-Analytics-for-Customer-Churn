# 🚀 End-to-End Customer Churn Prediction System

An industry-style **Machine Learning + MLOps project** for predicting customer churn and exposing the trained model through a production-ready API and web interface.

The objective of this project is not only to train a machine learning model, but to demonstrate the complete lifecycle of a real-world ML system:

**Data → EDA → Data Validation → Preprocessing → Feature Engineering → Feature Selection → Model Training → Hyperparameter Tuning → Evaluation → Explainability → Model Packaging → API → Frontend → Docker → Testing → Deployment → Monitoring**

---

## 📌 Project Overview

Customer churn prediction is a binary classification problem where the goal is to identify customers who are likely to stop using a subscription service.

In a real business environment, predicting churn can help a company:

* Identify customers at risk of leaving
* Prioritize retention campaigns
* Understand behavioral patterns
* Improve customer experience
* Reduce customer acquisition/replacement costs
* Support data-driven retention decisions

This project uses the **Predictive Analytics for Customer Churn** dataset available on Kaggle.

### Dataset Source

Kaggle:

https://www.kaggle.com/datasets/safrin03/predictive-analytics-for-customer-churn-dataset

The dataset contains customer subscription, payment, usage, support, and behavioral information. Kaggle describes files including:

* `train.csv`
* `test.csv`
* `data_descriptions.csv`

The available features include variables such as subscription type, payment method, paperless billing, content type, device, genre preference, account age, monthly charges, total charges, viewing behavior, support tickets, ratings, and watchlist size.

---

# 🎯 Business Problem

### Problem

A subscription-based company is losing customers but does not know which customers are most likely to churn.

The company wants a machine learning system that can:

1. Accept customer information.
2. Process the input automatically.
3. Predict whether the customer is likely to churn.
4. Return a churn probability.
5. Provide an interpretable prediction.
6. Expose the prediction through an API.
7. Provide a user-friendly web interface.
8. Run consistently across environments using Docker.

---

# 🎯 Machine Learning Objective

This is a **Supervised Binary Classification** problem.

### Target

The target variable represents whether a customer churns.

```text
Churn = 1 → Customer churned
Churn = 0 → Customer did not churn
```

### Model Output

The system should return:

```text
Prediction:
Churn / Not Churn

Probability:
0.00 – 1.00

Risk Level:
Low / Medium / High
```

Example:

```text
Customer Churn Prediction
-------------------------
Prediction: CHURN
Probability: 0.82
Risk Level: HIGH
```

---

# 🏗️ Project Architecture

The complete system follows this architecture:

```text
                    ┌─────────────────┐
                    │   Raw Dataset   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Data Validation │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │      EDA        │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Data Cleaning   │
                    └────────┬────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │    Feature Engineering     │
              │                             │
              │ Encoding                    │
              │ Transformation              │
              │ Scaling                     │
              │ Feature Creation             │
              │ Feature Selection            │
              │ Dimensionality Reduction    │
              └──────────────┬──────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ ML Preprocessing│
                    │    Pipeline     │
                    └────────┬────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │       Model Training        │
              │                             │
              │ Logistic Regression         │
              │ Decision Tree               │
              │ Random Forest               │
              │ Gradient Boosting           │
              │ Naive Bayes                 │
              │ Other suitable models       │
              └──────────────┬──────────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Hyperparameter      │
                  │ Tuning              │
                  │                     │
                  │ GridSearchCV        │
                  │ RandomizedSearchCV  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Model Evaluation    │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Explainability      │
                  │ SHAP / Feature      │
                  │ Importance          │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Model Serialization│
                  │       .pkl/.joblib  │
                  └──────────┬──────────┘
                             │
                ┌────────────┴─────────────┐
                │                          │
                ▼                          ▼
       ┌─────────────────┐       ┌─────────────────┐
       │   FastAPI       │       │   Streamlit     │
       │   REST API      │       │   Frontend      │
       └────────┬────────┘       └────────┬────────┘
                │                         │
                └────────────┬────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     Docker      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Deployment    │
                    └─────────────────┘
```

---

# 🧠 ML Lifecycle

The project is divided into several stages.

```text
1. Problem Definition
        ↓
2. Data Understanding
        ↓
3. Data Validation
        ↓
4. Exploratory Data Analysis
        ↓
5. Data Cleaning
        ↓
6. Train/Test Split
        ↓
7. Preprocessing
        ↓
8. Feature Engineering
        ↓
9. Feature Selection
        ↓
10. Baseline Models
        ↓
11. Model Comparison
        ↓
12. Hyperparameter Tuning
        ↓
13. Final Evaluation
        ↓
14. Explainability
        ↓
15. Model Packaging
        ↓
16. API Development
        ↓
17. Frontend Development
        ↓
18. Dockerization
        ↓
19. Testing
        ↓
20. Deployment
        ↓
21. Monitoring
```

---

# 📂 Project Structure

```text
customer-churn-mlops/
│
├── README.md
├── LICENSE
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── Makefile
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── data_descriptions.csv
│   │
│   ├── interim/
│   │
│   └── processed/
│       ├── X_train.csv
│       ├── X_valid.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_valid.csv
│       └── y_test.csv
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_data_cleaning.ipynb
│   ├── 04_preprocessing.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_baseline_models.ipynb
│   ├── 07_model_comparison.ipynb
│   ├── 08_hyperparameter_tuning.ipynb
│   ├── 09_model_evaluation.ipynb
│   └── 10_model_explainability.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   ├── validate_data.py
│   │   └── split_data.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── preprocessing_pipeline.py
│   │   ├── transformers.py
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── tune.py
│   │   └── predict.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── config.py
│   │   └── exceptions.py
│   │
│   └── monitoring/
│       ├── drift.py
│       └── performance.py
│
├── api/
│   ├── main.py
│   ├── schemas.py
│   ├── dependencies.py
│   └── routes/
│       └── prediction.py
│
├── frontend/
│   ├── app.py
│   ├── components.py
│   └── utils.py
│
├── models/
│   ├── preprocessing_pipeline.joblib
│   ├── churn_model.joblib
│   └── model_metadata.json
│
├── artifacts/
│   ├── figures/
│   ├── reports/
│   └── metrics/
│
├── tests/
│   ├── test_data.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   ├── test_api.py
│   └── test_prediction.py
│
├── configs/
│   └── config.yaml
│
└── .github/
    └── workflows/
        └── ci.yml
```

---

# 📊 Dataset Understanding

The dataset contains customer-level subscription and behavioral information.

Important feature groups include:

### Customer Information

```text
CustomerID
Gender
```

### Subscription Information

```text
SubscriptionType
AccountAge
MonthlyCharges
TotalCharges
```

### Payment Information

```text
PaymentMethod
PaperlessBilling
```

### Content Behavior

```text
ContentType
GenrePreference
ViewingHoursPerWeek
AverageViewingDuration
ContentDownloadsPerMonth
WatchlistSize
```

### Device Information

```text
MultiDeviceAccess
DeviceRegistered
```

### Customer Support

```text
SupportTicketsPerMonth
```

### Customer Satisfaction

```text
UserRating
```

The exact columns and data types should always be verified from the downloaded dataset rather than assumed from documentation.

---

# 🔎 Phase 1 — Data Understanding

Objectives:

* Load raw data.
* Understand dataset dimensions.
* Inspect columns.
* Identify target variable.
* Identify numerical variables.
* Identify categorical variables.
* Identify identifier columns.
* Check data types.
* Check unique values.
* Understand target distribution.

Questions to answer:

```text
How many customers are present?

How many features are available?

What is the target variable?

Is the target balanced?

Which columns are numerical?

Which columns are categorical?

Are there duplicate customers?

Are there suspicious columns?

Are there missing values?
```

---

# 📈 Phase 2 — Exploratory Data Analysis

EDA should answer business questions rather than only generate graphs.

## Univariate Analysis

Analyze:

* Numerical distributions
* Categorical distributions
* Target distribution
* Outliers
* Skewness

Visualizations:

```text
Histogram
Boxplot
Countplot
Density plot
```

---

## Bivariate Analysis

Analyze relationships between:

```text
Feature vs Churn
```

Examples:

```text
SubscriptionType vs Churn

MonthlyCharges vs Churn

AccountAge vs Churn

SupportTicketsPerMonth vs Churn

ViewingHoursPerWeek vs Churn

UserRating vs Churn
```

---

## Multivariate Analysis

Investigate:

* Correlation
* Feature interactions
* Numerical feature relationships
* Categorical combinations
* Potential multicollinearity

Visualizations:

```text
Correlation Heatmap
Pairplot
Grouped Bar Charts
Boxplots
```

---

# 🧹 Phase 3 — Data Cleaning

Data cleaning should be performed systematically.

Possible operations:

```text
Missing-value handling
Duplicate detection
Invalid-value detection
Incorrect datatype correction
Whitespace cleaning
Category normalization
Outlier investigation
Identifier handling
```

Example:

```text
" Premium "
"Premium"
"premium"
```

should be normalized when appropriate.

---

# ⚠️ Data Leakage Prevention

Data leakage is one of the most important concepts in this project.

The following must NOT happen:

```text
Test data → preprocessing fitting

Test data → feature selection

Test data → imputation statistics

Test data → scaler fitting

Test data → hyperparameter tuning
```

Correct approach:

```text
Training Data
      ↓
Fit preprocessing
      ↓
Fit feature engineering
      ↓
Fit model
      ↓
Validation/Test Data
      ↓
Transform only
      ↓
Predict
```

All learned preprocessing operations should be fitted only on training data.

---

# ✂️ Phase 4 — Train/Validation/Test Split

Recommended strategy:

```text
Train       → 70%
Validation  → 15%
Test        → 15%
```

For a binary classification problem, use:

```python
stratify=y
```

to preserve the class distribution.

The final test set must remain untouched until final evaluation.

---

# ⚙️ Phase 5 — Preprocessing

Preprocessing and feature engineering are related but should be treated as separate responsibilities.

## Preprocessing

Typical preprocessing operations:

### Numerical Features

```text
Missing-value imputation
Scaling
Optional transformation
```

Example:

```text
Median Imputation
        ↓
StandardScaler
```

### Categorical Features

```text
Missing-value imputation
        ↓
One-Hot Encoding
```

or another appropriate encoding strategy depending on cardinality and model requirements.

---

# 🧠 Phase 6 — Feature Engineering

Feature engineering creates better representations of the original information.

Potential features for this dataset should be created only when they make business/statistical sense.

Examples:

### Engagement Features

```text
EngagementScore
ViewingIntensity
DownloadIntensity
```

### Customer Support Features

```text
SupportPressure
```

### Revenue Features

```text
EstimatedMonthlyValue
```

### Usage Features

```text
ViewingPerMonth
```

Important:

**Do not create features simply to increase model accuracy.**

Every engineered feature should have a logical explanation.

---

# 🎯 Feature Selection

Possible techniques:

```text
Correlation analysis
Variance threshold
Mutual information
SelectKBest
Recursive Feature Elimination
Model-based feature importance
L1 regularization
```

Feature selection must be performed inside the training workflow where appropriate to prevent leakage.

---

# 📉 Dimensionality Reduction

Dimensionality reduction can be explored as an experiment.

Possible technique:

```text
PCA
```

However, PCA should not automatically be included in the final production pipeline.

For example:

```text
Experiment A
Raw features → Model

Experiment B
Engineered features → Model

Experiment C
Engineered features → PCA → Model
```

Compare the results.

If PCA does not provide meaningful benefit, document that decision.

---

# 🔗 Phase 7 — ML Pipeline

The production model should use a reproducible pipeline.

Conceptually:

```text
Raw Input
   ↓
Column Selection
   ↓
Missing Value Imputation
   ↓
Encoding
   ↓
Scaling
   ↓
Feature Engineering
   ↓
Feature Selection
   ↓
Classifier
```

Example architecture:

```python
Pipeline([
    ("preprocessor", preprocessor),
    ("feature_engineering", feature_engineering),
    ("feature_selection", feature_selector),
    ("classifier", model)
])
```

For mixed numerical and categorical data, use:

```text
ColumnTransformer
```

This ensures that the same transformations are applied during:

```text
Training
Validation
Testing
API prediction
```

---

# 🤖 Phase 8 — Baseline Models

Start with simple models before advanced tuning.

Recommended candidates:

```text
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Naive Bayes
5. Gradient Boosting
```

Optional:

```text
XGBoost
LightGBM
CatBoost
```

if computational resources and project requirements permit.

---

# 📊 Model Comparison

Create a model comparison table.

Example structure:

| Model               | Accuracy | Precision | Recall |  F1 | ROC-AUC |
| ------------------- | -------: | --------: | -----: | --: | ------: |
| Logistic Regression |      TBD |       TBD |    TBD | TBD |     TBD |
| Decision Tree       |      TBD |       TBD |    TBD | TBD |     TBD |
| Random Forest       |      TBD |       TBD |    TBD | TBD |     TBD |
| Naive Bayes         |      TBD |       TBD |    TBD | TBD |     TBD |
| Gradient Boosting   |      TBD |       TBD |    TBD | TBD |     TBD |

**Do not select the final model based only on accuracy.**

For churn prediction, evaluate:

```text
Precision
Recall
F1-score
ROC-AUC
PR-AUC
Confusion Matrix
Calibration
```

---

# 🎛️ Phase 9 — Hyperparameter Tuning

After baseline comparison, tune promising models.

Methods:

```text
GridSearchCV
RandomizedSearchCV
```

Optional advanced approach:

```text
Optuna
```

Example parameters:

### Logistic Regression

```text
C
penalty
solver
class_weight
```

### Random Forest

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
max_features
class_weight
```

### Gradient Boosting

```text
n_estimators
learning_rate
max_depth
subsample
```

Use cross-validation.

Example:

```text
StratifiedKFold
```

---

# 🔬 Phase 10 — Model Evaluation

Final evaluation must be performed on the untouched test set.

## Confusion Matrix

```text
                 Predicted
                 0       1

Actual 0        TN      FP

Actual 1        FN      TP
```

---

## Accuracy

Measures overall correct predictions.

```text
Accuracy =
(TP + TN) / (TP + TN + FP + FN)
```

---

## Precision

Of customers predicted as churners, how many actually churned?

```text
Precision =
TP / (TP + FP)
```

---

## Recall

Of customers who actually churned, how many did the model identify?

```text
Recall =
TP / (TP + FN)
```

---

## F1 Score

Balances precision and recall.

```text
F1 =
2 × Precision × Recall
/
(Precision + Recall)
```

---

## ROC-AUC

Measures the model's ability to distinguish between churn and non-churn customers across classification thresholds.

---

## PR-AUC

Particularly useful when the positive class is relatively rare.

---

# 🎚️ Probability Threshold Optimization

Do not automatically assume:

```text
threshold = 0.50
```

Test different thresholds:

```text
0.30
0.40
0.50
0.60
0.70
```

Then analyze:

```text
Precision
Recall
F1
Business cost
```

The final threshold should be a documented business/modeling decision.

---

# 💰 Business Cost Analysis

A production churn model should connect ML metrics to business consequences.

Example:

```text
False Negative:
High-risk customer is missed.

False Positive:
Company spends retention resources on a customer who would not have churned.
```

Create a simple cost matrix:

| Prediction | Actual   | Business Impact              |
| ---------- | -------- | ---------------------------- |
| Churn      | Churn    | Correct intervention         |
| Churn      | No Churn | Unnecessary intervention     |
| No Churn   | Churn    | Missed retention opportunity |
| No Churn   | No Churn | Correct decision             |

This converts the project from a pure ML exercise into a business-oriented ML system.

---

# 🔍 Phase 11 — Model Explainability

A production ML system should provide some explanation for predictions.

Possible approaches:

```text
Feature Importance
Permutation Importance
SHAP
```

For tree-based models:

```text
SHAP TreeExplainer
```

For individual predictions:

```text
Customer Prediction
       ↓
Why did the model predict churn?
       ↓
Top contributing features
```

Example output:

```text
Prediction: CHURN

Probability: 0.84

Important factors:

+ High support tickets
+ High monthly charges
- Long account age
+ Low user rating
```

These explanations must be presented as **model contributions**, not causal claims.

---

# 📦 Phase 12 — Model Serialization

Save the complete inference pipeline.

Example:

```text
models/
│
├── churn_model.joblib
├── preprocessing_pipeline.joblib
└── model_metadata.json
```

Prefer saving the complete preprocessing + model pipeline together when possible.

This avoids inconsistencies between training and production inference.

---

# 🌐 Phase 13 — REST API

Use **FastAPI** to expose the model.

Architecture:

```text
Client
  ↓
HTTP Request
  ↓
FastAPI
  ↓
Input Validation
  ↓
Preprocessing Pipeline
  ↓
ML Model
  ↓
Prediction
  ↓
JSON Response
```

---

# 📡 API Endpoints

## Health Check

```http
GET /health
```

Response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## Prediction

```http
POST /predict
```

Example request:

```json
{
  "subscription_type": "Premium",
  "payment_method": "Credit Card",
  "paperless_billing": "Yes",
  "content_type": "Movies",
  "multi_device_access": "Yes",
  "device_registered": "Smart TV",
  "genre_preference": "Drama",
  "gender": "Male",
  "parental_control": "No",
  "subtitles_enabled": "Yes",
  "account_age": 12,
  "monthly_charges": 15.5,
  "total_charges": 186.0,
  "viewing_hours_per_week": 12,
  "support_tickets_per_month": 2,
  "average_viewing_duration": 45,
  "content_downloads_per_month": 8,
  "user_rating": 3,
  "watchlist_size": 20
}
```

Example response:

```json
{
  "prediction": 1,
  "prediction_label": "Churn",
  "probability": 0.82,
  "risk_level": "High"
}
```

The exact request schema must be generated from the final training features.

---

# 🖥️ Phase 14 — Frontend

Use **Streamlit** for the user interface.

The frontend should allow users to enter customer information through:

```text
Dropdowns
Number inputs
Radio buttons
Sliders
Checkboxes
```

Example:

```text
Customer Churn Prediction

Subscription Type
[ Premium ▼ ]

Monthly Charges
[ 15.5 ]

Account Age
[ 12 ]

Support Tickets
[ 2 ]

User Rating
[ 3 ]

        [ Predict Churn ]

--------------------------------

Prediction: HIGH RISK

Churn Probability: 82%

Risk Level: HIGH
```

---

# 🔐 API Key & Environment Variables

Secrets should never be hardcoded.

Do NOT:

```python
API_KEY = "my-secret-key"
```

Instead use:

```text
.env
```

Example:

```env
API_KEY=your_secret_key
MODEL_PATH=models/churn_model.joblib
API_URL=http://localhost:8000
```

Commit only:

```text
.env.example
```

Never commit:

```text
.env
```

Add it to `.gitignore`.

---

# 🐳 Phase 15 — Docker

The application should be containerized so that it can run consistently across environments.

Example architecture:

```text
Docker
│
├── FastAPI Container
│
└── Streamlit Container
```

Optional:

```text
Docker Compose
```

Architecture:

```text
                 Docker Compose
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
    FastAPI API               Streamlit UI
       :8000                     :8501
          │                         │
          └────────────┬────────────┘
                       │
                 ML Pipeline
                       │
                 Trained Model
```

---

# 🧪 Phase 16 — Testing

Production ML projects need tests.

## Data Tests

Check:

```text
Required columns exist
Expected datatypes
Target values are valid
No unexpected nulls
No duplicate IDs
```

---

## Preprocessing Tests

Check:

```text
Pipeline executes
Expected feature count
Categorical encoding works
Missing values are handled
```

---

## Model Tests

Check:

```text
Model loads
Prediction is generated
Probability is between 0 and 1
Prediction belongs to expected classes
```

---

## API Tests

Test:

```text
GET /health

POST /predict

Invalid input

Missing input

Incorrect datatype

Out-of-range values
```

Use:

```text
pytest
```

---

# 🔄 Phase 17 — CI/CD

Use GitHub Actions for continuous integration.

Example workflow:

```text
Git Push
   ↓
GitHub Actions
   ↓
Install Dependencies
   ↓
Lint
   ↓
Run Unit Tests
   ↓
Build Docker Image
   ↓
API Tests
   ↓
Success / Failure
```

Example:

```text
.github/
└── workflows/
    └── ci.yml
```

---

# 📋 Code Quality

Use:

```text
PEP8
Type hints
Docstrings
Modular functions
Logging
Exception handling
Configuration management
```

Recommended tools:

```text
ruff
black
pytest
```

---

# 📝 Logging

Production systems should use logging instead of random `print()` statements.

Log important events:

```text
Application started
Model loaded
Prediction requested
Prediction completed
Invalid input
API error
```

Avoid logging sensitive customer information.

---

# 📊 Model Monitoring

After deployment, model performance can change.

Monitor:

### Data Drift

Check whether production data differs from training data.

Example:

```text
Training MonthlyCharges distribution
        vs
Production MonthlyCharges distribution
```

Possible techniques:

```text
PSI
KS Test
Distribution comparison
```

---

# 📉 Prediction Monitoring

Monitor:

```text
Prediction distribution
Churn probability distribution
Average probability
Input feature distributions
API latency
Error rate
```

---

# 🚨 Model Retraining Strategy

A future production workflow could be:

```text
Production Data
      ↓
Data Validation
      ↓
Drift Detection
      ↓
Performance Monitoring
      ↓
Retraining Trigger
      ↓
New Model Training
      ↓
Evaluation
      ↓
Model Approval
      ↓
Deployment
```

---

# 🧪 Experiment Tracking

For an advanced MLOps implementation, use an experiment-tracking tool such as:

```text
MLflow
```

Track:

```text
Model name
Hyperparameters
Training dataset version
Metrics
Artifacts
Model version
```

Example:

```text
Experiment
│
├── Logistic Regression
│   ├── Parameters
│   ├── Metrics
│   └── Model
│
├── Random Forest
│   ├── Parameters
│   ├── Metrics
│   └── Model
│
└── Gradient Boosting
    ├── Parameters
    ├── Metrics
    └── Model
```

---

# 📦 Reproducibility

The project should be reproducible.

Record:

```text
Python version
Package versions
Random seeds
Dataset version
Model configuration
Feature configuration
Training configuration
```

Example:

```text
Python 3.x
scikit-learn
pandas
numpy
fastapi
streamlit
joblib
pytest
```

Exact versions should be generated from the working environment rather than guessed.

---

# 🔐 Security Considerations

The API should validate incoming data.

Protect against:

```text
Invalid input
Unexpected values
Oversized requests
Missing fields
Incorrect datatypes
Unauthorized API access
```

Never expose:

```text
API secrets
Environment variables
Internal paths
Stack traces
Customer-sensitive information
```

---

# 📈 Model Card

The project should contain a model card describing:

## Model

```text
Model name:
Version:
Training date:
Dataset:
Target:
```

## Performance

```text
Accuracy:
Precision:
Recall:
F1:
ROC-AUC:
PR-AUC:
```

These values should be filled automatically from the final experiment.

## Limitations

Possible limitations include:

* Dataset may not represent a real company's complete customer population.
* Historical/simulated relationships may not generalize to another business.
* Predictions indicate statistical risk rather than guaranteed future behavior.
* Model performance depends on input data quality.
* Feature importance does not automatically imply causality.

---

# 🧭 Complete Development Roadmap

## Phase 1 — Foundation

```text
[ ] Create GitHub repository
[ ] Create project structure
[ ] Create virtual environment
[ ] Install dependencies
[ ] Download dataset
[ ] Create configuration
```

---

## Phase 2 — Data

```text
[ ] Load data
[ ] Validate schema
[ ] Inspect missing values
[ ] Inspect duplicates
[ ] Inspect datatypes
[ ] Identify target
[ ] Split train/test
```

---

## Phase 3 — EDA

```text
[ ] Univariate analysis
[ ] Bivariate analysis
[ ] Multivariate analysis
[ ] Churn distribution
[ ] Outlier analysis
[ ] Correlation analysis
[ ] Business insights
```

---

## Phase 4 — Preprocessing

```text
[ ] Missing-value strategy
[ ] Numerical preprocessing
[ ] Categorical preprocessing
[ ] Encoding
[ ] Scaling
[ ] ColumnTransformer
```

---

## Phase 5 — Feature Engineering

```text
[ ] Domain-based features
[ ] Interaction features
[ ] Ratio features
[ ] Aggregation features
[ ] Feature selection
[ ] Optional dimensionality reduction
```

---

## Phase 6 — Modeling

```text
[ ] Logistic Regression
[ ] Decision Tree
[ ] Random Forest
[ ] Naive Bayes
[ ] Gradient Boosting
[ ] Model comparison
```

---

## Phase 7 — Optimization

```text
[ ] Cross-validation
[ ] Hyperparameter tuning
[ ] Threshold optimization
[ ] Calibration analysis
[ ] Final model selection
```

---

## Phase 8 — Explainability

```text
[ ] Feature importance
[ ] Permutation importance
[ ] SHAP
[ ] Individual prediction explanation
```

---

## Phase 9 — Production

```text
[ ] Save pipeline
[ ] Save model
[ ] Build FastAPI
[ ] Create Pydantic schemas
[ ] Build prediction endpoint
[ ] Add health endpoint
```

---

## Phase 10 — Frontend

```text
[ ] Streamlit UI
[ ] Input form
[ ] API connection
[ ] Prediction result
[ ] Probability
[ ] Risk level
[ ] Explanation
```

---

## Phase 11 — MLOps

```text
[ ] Logging
[ ] Testing
[ ] Configuration
[ ] MLflow
[ ] Docker
[ ] Docker Compose
[ ] GitHub Actions
[ ] CI pipeline
```

---

## Phase 12 — Monitoring

```text
[ ] Data drift
[ ] Prediction drift
[ ] API latency
[ ] Error monitoring
[ ] Model performance monitoring
[ ] Retraining strategy
```

---

# 🏆 Final Production Architecture

The final system should look like:

```text
                         USER
                          │
                          ▼
                 ┌────────────────┐
                 │   Streamlit UI │
                 └───────┬────────┘
                         │
                         │ HTTP
                         ▼
                 ┌────────────────┐
                 │    FastAPI     │
                 │      API       │
                 └───────┬────────┘
                         │
                         ▼
                 ┌────────────────┐
                 │ Input Validation│
                 └───────┬────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Preprocessing + FE   │
              │      Pipeline        │
              └──────────┬───────────┘
                         │
                         ▼
                 ┌────────────────┐
                 │ Churn Model    │
                 └───────┬────────┘
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
       Prediction               Probability
             │                       │
             └───────────┬───────────┘
                         ▼
                   Risk Analysis
                         │
                         ▼
                  API JSON Response
                         │
                         ▼
                    Streamlit UI


              ───── MLOps Layer ─────

       GitHub → CI/CD → Docker → Deployment
                     │
                     ▼
                 Monitoring
                     │
                     ▼
               Drift Detection
                     │
                     ▼
               Model Retraining
```

---

# 💼 Resume Project Description

### Customer Churn Prediction — End-to-End ML & MLOps System

**Tech Stack:** Python, Pandas, NumPy, Scikit-learn, FastAPI, Streamlit, Docker, Pytest, MLflow, GitHub Actions

* Built an end-to-end customer churn prediction system covering EDA, data validation, preprocessing, feature engineering, model training, cross-validation, and hyperparameter optimization.
* Developed reproducible Scikit-learn preprocessing and modeling pipelines to prevent training-serving skew and data leakage.
* Compared multiple classification algorithms using Precision, Recall, F1-score, ROC-AUC, PR-AUC, and confusion-matrix analysis.
* Implemented model explainability using feature importance/SHAP to understand individual churn predictions.
* Exposed the trained ML pipeline through a FastAPI REST API and developed a Streamlit interface for interactive predictions.
* Containerized the application using Docker and implemented automated testing/CI workflows.
* Designed a monitoring architecture for data drift, prediction drift, API performance, and future model retraining.

---

# 🧑‍💻 Skills Demonstrated

This project demonstrates:

### Python

```text
Python
OOP
Type Hints
Exception Handling
Logging
Package Structure
```

### Data Science

```text
Pandas
NumPy
EDA
Statistics
Visualization
Data Cleaning
```

### Machine Learning

```text
Classification
Logistic Regression
Decision Trees
Random Forest
Naive Bayes
Gradient Boosting
Cross Validation
Hyperparameter Tuning
Feature Engineering
Feature Selection
Dimensionality Reduction
Model Evaluation
Threshold Optimization
```

### MLOps

```text
ML Pipelines
Experiment Tracking
Model Serialization
Model Versioning
Testing
Docker
CI/CD
Monitoring
Drift Detection
```

### Backend

```text
FastAPI
REST API
Pydantic
API Validation
JSON
Authentication/API Key Concepts
```

### Frontend

```text
Streamlit
Interactive Forms
API Integration
Prediction Dashboard
```

### DevOps

```text
Git
GitHub
Docker
GitHub Actions
Environment Variables
```

---

# 📌 Important Design Principles

This project follows several industry principles.

### 1. Notebook ≠ Production System

Notebooks are used for:

```text
Exploration
Experiments
Visualization
Research
```

Production code belongs inside:

```text
src/
api/
tests/
```

---

### 2. Don't Train Inside the API

Bad architecture:

```text
API → Load data → Train model → Predict
```

Correct architecture:

```text
Training Pipeline
      ↓
Trained Model
      ↓
Saved Artifact
      ↓
API
      ↓
Prediction
```

---

### 3. Don't Duplicate Preprocessing

Avoid:

```text
Notebook preprocessing
+
API preprocessing
```

Instead:

```text
Single preprocessing pipeline
             ↓
Training + Inference
```

---

### 4. Don't Use Test Data During Development

The test dataset should remain untouched until the final evaluation.

---

### 5. Don't Chase Accuracy

The objective is not:

```text
Highest Accuracy
```

The objective is:

```text
Reliable
Reproducible
Interpretable
Business-useful
Deployable
Maintainable
```

---

# ⚠️ Dataset & Project Limitations

This project is primarily an educational and portfolio implementation.

The dataset is publicly available through Kaggle and is described as an anonymized subscription-service dataset for churn analysis. Kaggle notes that it was used for educational/research purposes.

Therefore:

* Model performance should not be interpreted as production performance for a real company.
* Business decisions should not rely solely on the model.
* Feature importance should not be interpreted as causal relationships.
* Real production deployment would require stronger data governance, privacy controls, monitoring, retraining policies, and validation on real business data.

---

# 🚀 Future Improvements

Possible future versions:

```text
[ ] MLflow model registry
[ ] DVC dataset versioning
[ ] Airflow/Prefect pipeline orchestration
[ ] Cloud deployment
[ ] Redis caching
[ ] PostgreSQL database
[ ] Authentication
[ ] Role-based access
[ ] Advanced monitoring
[ ] Automated retraining
[ ] Model registry
[ ] A/B testing
[ ] Feature store
[ ] Real-time prediction
```

---

# ⭐ Project Goal

The goal of this project is not simply to say:

> "I trained a churn prediction model."

The goal is to demonstrate:

> **"I can take a machine learning problem from raw data to a reproducible, tested, explainable and deployable ML system."**

That is the core idea behind this portfolio project.

---

# 📚 References

* Kaggle — Predictive Analytics for Customer Churn Dataset:
  https://www.kaggle.com/datasets/safrin03/predictive-analytics-for-customer-churn-dataset

* Scikit-learn documentation:
  https://scikit-learn.org/

* FastAPI documentation:
  https://fastapi.tiangolo.com/

* Streamlit documentation:
  https://docs.streamlit.io/

* Docker documentation:
  https://docs.docker.com/

---

# 👨‍💻 Author

**Ajit Kumar**

Aspiring Data Scientist / Machine Learning Engineer

India

---

## ⭐ If this project helps you learn something, consider giving the repository a star.
