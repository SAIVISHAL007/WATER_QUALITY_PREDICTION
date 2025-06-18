# WEEK-1
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

1. Clone this repository and place your dataset (e.g., `water_data.csv`) in the project folder.
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

## 📄 License

This project is licensed under the MIT License.

---

**Contributions and feedback are welcome!**
