# 🏚️ Predicting House Prices Using Linear Regression, Random Forest & Gradient Boosting 💰

This project applies advanced regression techniques to the popular [House Prices: Advanced Regression Techniques] dataset from Kaggle. The goal is to predict housing prices based on various structural and locational features of residential homes in Ames, Iowa.
## 📌 Tools & Libraries
Python (Pandas, NumPy, Matplotlib, Seaborn)

Scikit-learn (Pipelines, GridSearchCV, Models)

Statsmodels (VIF analysis)

Kaggle (Dataset-https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

---

## 🧠 Project Motivation

Accurate house price prediction is crucial for real estate agents, buyers, and sellers. It involves analyzing a variety of features like square footage, number of rooms, quality ratings, and neighborhood information. This project focuses on building and comparing predictive models, performing feature selection, and optimizing performance using hyperparameter tuning and ensemble methods.

---

## 📂 Dataset Overview

- `train.csv`: 1460 rows × 81 columns of training data with sale prices.
- `test.csv`: 1459 rows × 80 columns (excluding `SalePrice`).
- `sample_submission.csv`: Submission template.
- Features include:
  - **Numerical**: Lot area, Year built, Living area, etc.
  - **Categorical**: Neighborhood, Building type, etc.

---

## 🔍 Workflow

### 1. **Exploratory Data Analysis (EDA)**
- Checked for missing values and duplicates.
- Visualized distributions and relationships with `SalePrice`.
- Correlation heatmaps and pairwise comparisons to identify strong predictors.

### 2. **Data Preprocessing**
- Missing values filled using **median** (numerical) and **mode** (categorical).
- One-hot encoding for categorical features.
- Feature scaling using `StandardScaler`.

### 3. **Feature Engineering & Selection**
- Identified top correlated features with `SalePrice`.
- Checked for **multicollinearity** using **Variance Inflation Factor (VIF)**.
- Selected final features based on correlation, VIF, and domain understanding.

### 4. **Modeling**
Built and compared the following models using pipelines:
- 🔹 **Linear Regression**
- 🔹 **Random Forest Regressor**
- 🔹 **Gradient Boosting Regressor**
- 🔹 **Ridge Regression**

Metrics used:
- **RMSE (Root Mean Squared Error)**
- **R² Score (Coefficient of Determination)**

### 5. **Hyperparameter Tuning**
Used `GridSearchCV` to fine-tune:
- Gradient Boosting
- Random Forest
- Linear Regression (fit intercept)

### 6. **Ensemble Model**
Created a blended model by averaging predictions from:
- Best Linear Regression
- Best Random Forest
- Tuned Gradient Boosting

This ensemble yielded strong generalization on validation data.

---

## 🔢 Final Results

| Model                  | RMSE    | R² Score |
|------------------------|---------|----------|
| Linear Regression      | ~37,000 | ~0.85    |
| Random Forest          | ~34,000 | ~0.88    |
| Gradient Boosting      | ~31,000 | ~0.90    |
| **Ensemble Model**     | ~31,200 | **0.87+** |

---

## 📊 Feature Importance

Top predictive features based on Gradient Boosting:
- `OverallQual`
- `GrLivArea`
- `GarageArea`
- `TotalBsmtSF`
- `YearBuilt`

---

## 📁 Submission

Predictions for the test set were made using the final ensemble model and saved to `submission.csv`:





