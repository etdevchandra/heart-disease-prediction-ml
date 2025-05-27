# Heart Disease Prediction: A Machine Learning Approach

This project applies machine learning to predict heart disease using clinical and demographic data. It demonstrates data preprocessing, feature engineering, algorithm comparison, evaluation, and ethical considerations to build an interpretable, high-performance prediction model.

---

## Project Overview

- **Goal:** Accurately predict heart disease presence based on patient attributes.
- **Tech Stack:** Python, Jupyter Notebook, scikit-learn, pandas, matplotlib, seaborn, imbalanced-learn.
- **Dataset:** Consolidated from multiple Kaggle heart disease sources.

---

## Project Structure

- **/data/** — Contains the heart disease dataset.
- **/notebooks/** — Jupyter Notebooks for preprocessing, EDA, and model experimentation.
- **/report/** — Full project report with analysis and results.

---

## Dataset

- **Source:** Combined from Cleveland, Hungary, Switzerland, Long Beach VA, and Statlog datasets via Kaggle.
- **Rows:** 1,190 patient records
- **Features:** 11 clinical and demographic variables

---

## How to Run

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/heart-disease-prediction-ml.git
    cd heart-disease-prediction-ml
    ```
2. **Install required Python libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *(If you don’t have a `requirements.txt`, install manually: `pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn`)*

3. **Open and run the notebooks:**
    - Start with `01-data-preprocessing-eda.ipynb`
    - Then run `02-model-selection-and-evaluation.ipynb`

4. **Review results and visualizations.**

---

## Highlights

- **Data Preprocessing:** Handled missing values, outliers, and feature scaling (z-score normalization).
- **Feature Engineering:** One-hot encoding for categorical variables, SMOTE for class imbalance, visualization of demographic bias.
- **Modeling:** Compared Logistic Regression, Random Forest, Decision Tree, K-Nearest Neighbors (KNN), and Multi-Layer Perceptron (MLP/ANN).
- **Evaluation:** Used cross-validation, confusion matrix, F1-score, ROC-AUC, and feature importance metrics.
- **Interpretability & Ethics:** Emphasized transparency, clinical relevance, and fairness (especially for gender imbalance). Includes full ethical review.

---

## Results

- **Best Model:** Random Forest
- **Key Metrics:**
    - **F1-score:** 0.91
    - **ROC-AUC:** 0.96
    - **Recall & Precision:** Both > 0.90
- **Key Predictors Identified:** ST slope, chest pain type, oldpeak, exercise-induced angina, and maximum heart rate.

---

## Full Report

For a comprehensive breakdown of methods, analysis, results, and ethical discussion, see:
- `report/heart-disease-prediction-project-report.pdf`

---

## Author

**Enoshan Devchandra**  
BSc (Hons) Software Engineering  
University of Central Lancashire  
Email: [enoshtim@gmail.com](mailto:enoshtim@gmail.com)

---

## Contributing

Pull requests are welcome! For significant changes, please open an issue first to discuss what you would like to change or improve.

---

## Acknowledgments

- [Kaggle](https://www.kaggle.com/datasets) for open-source heart disease datasets
- Scikit-learn and open-source ML community for tools and tutorials

---
