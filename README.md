# WEEK-1 & WEEK-2
# Water Quality Multi-Output Regression

This project demonstrates a complete machine learning workflow for predicting multiple water quality pollutants using a Random Forest regressor. The workflow includes data exploration, preprocessing, feature engineering, model training, evaluation, and visualization.

## 📁 Dataset

The dataset contains water quality measurements from various locations and dates, with columns such as:

- `id`: Location identifier
- `date`: Measurement date
- `NH4`, `BSK5`, `Suspended`, `O2`, `NO3`, `NO2`, `SO4`, `PO4`, `CL`: Various chemical and physical water quality indicators

##  Workflow

1. **Data Exploration**
   - View dataset info, summary statistics, and missing values.
2. **Preprocessing**
   - Convert date columns, extract year and month, sort data.
   - Handle missing values using median imputation.
3. **Feature Engineering**
   - Use location, temporal, and chemical features for prediction.
4. **Modeling**
   - Train a multi-output Random Forest regressor to predict several pollutants at once.
5. **Evaluation**
   - Compute R² score and Mean Squared Error (MSE) for each pollutant.
   - Visualize R² scores and feature importances.

## 📊 Example Outputs

- **Metrics Table:**  
  Shows R² and MSE for each pollutant.
- **Bar Plot:**  
  Visualizes R² scores for all pollutants.
- **Feature Importance Plot:**  
  Displays which features are most important for predicting a selected pollutant.

## 🛠 Usage

1. Clone this repository and place your dataset ((e.g., `afa2e701598d20110228.csv`) a.k.a water_data_set . csv)  in the project folder.
2. Open the notebook or script in [Google Colab](https://colab.research.google.com/) or Jupyter.
3. Run the cells step by step.  
   *(If using Colab, upload your CSV when prompted.)*

## 📦 Requirements

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
# 🌊 Week 2 - Water Quality Prediction using Random Forest Regressor

## 📅 Internship Week: 23 June 2025  
**Mentor-Guided Task:** Enhancing pollutant prediction using Random Forest and station/year-based features.

---

## 🧠 Objective

The aim of Week 2 is to improve our predictive model for water pollutants by:
- Cleaning and encoding the data properly.
- Using **station ID** and **year** as predictors.
- Training a **MultiOutput Random Forest Regressor**.
- Evaluating performance using R² Score and Mean Squared Error (MSE).
- Saving the trained model and feature structure for future use.

---

## 🗂 Dataset Description

Source CSV file: `afa2e701598d20110228.csv`  
Total Records: 2861  
Key columns used:
- Features: `id`, `year`
- Targets: `O2`, `NO3`, `NO2`, `SO4`, `PO4`, `CL`

---

## 🧹 Preprocessing Steps

- Converted the `date` column into datetime format.
- Extracted `year` and `month` from the date.
- Removed rows with missing target pollutant values using `dropna()`.
- Applied **One-Hot Encoding** to the `id` (station) column.
- Aligned prediction input with training feature columns.

---

## 🏗️ Model Used

- **Model Type**: MultiOutput Regressor
- **Base Estimator**: RandomForestRegressor (100 trees, random_state=42)
- **Library**: `scikit-learn`

---

## 📈 Model Evaluation

Evaluated using R² Score and Mean Squared Error on the test dataset:

| Pollutant | R² Score | MSE        |
|-----------|----------|------------|
| O2        | -0.0167  | 22.22      |
| NO3       | 0.5162   | 18.15      |
| NO2       | -78.42   | 10.61      |
| SO4       | 0.4118   | 2412.14    |
| PO4       | 0.3221   | 0.38       |
| CL        | 0.7358   | 34882.82   |

---

## 📊 Prediction Example

For input:  
`station_id = '22'` and `year = 2024`

**Predicted Pollutant Levels**:
- O2: 12.60  
- NO3: 6.88  
- NO2: 0.13  
- SO4: 143.08  
- PO4: 0.50  
- CL: 67.33

---

## 💾 Model Persistence

The following files are saved for future use:
- `pollution_model.pkl`: Trained MultiOutput Random Forest model
- `model_columns.pkl`: Structure of input features used for prediction

Use `joblib.load()` to reload these in future weeks.

---

## ✅ Summary

Week 2 focused on improving prediction performance by simplifying inputs and ensuring a production-ready pipeline. While some pollutants like NO2 still need improvement, the overall model generalizes reasonably well and is now deployment-ready.

---


## 📄 License

This project is licensed under the MIT License.

---

**Contributions and feedback are welcome!**
