# Heart Disease Prediction Project

This project contains multiple machine learning notebooks for predicting heart disease from the `heart.csv` dataset. It compares three classifiers - Gaussian Naive Bayes, Logistic Regression, and Random Forest - in both initial and optimized notebook versions.

The notebooks cover common preprocessing and evaluation steps such as:

- Loading the dataset from `heart.csv`
- Encoding categorical variables
- Splitting data into training and test sets
- Handling class imbalance with SMOTE
- Scaling features where needed
- Training and tuning classification models
- Evaluating results with accuracy, classification reports, confusion matrices, and learning curves

## Dataset

The dataset includes 12 columns:

- `Age`
- `Sex`
- `ChestPainType`
- `RestingBP`
- `Cholesterol`
- `FastingBS`
- `RestingECG`
- `MaxHR`
- `ExerciseAngina`
- `Oldpeak`
- `ST_Slope`
- `HeartDisease` target label

The target variable is `HeartDisease`, where `0` means no heart disease and `1` means heart disease.

## Notebook Overview

### 1. Gaussian Naive Bayes

- [Gussian_Naive_Bayes_Algorithm_Initial.ipynb](Gussian_Naive_Bayes_Algorithm_Initial.ipynb)
  - Builds a Gaussian Naive Bayes heart disease classifier
  - Uses SMOTE to balance the training data
  - Evaluates confusion matrix and learning curve
  - Reported test accuracy: `0.8478`

- [Gussian_Naive_Bayes_Algorithm_Optimized.ipynb](Gussian_Naive_Bayes_Algorithm_Optimized.ipynb)
  - Uses an `imblearn` pipeline with `StandardScaler`, `SMOTE`, and `GaussianNB`
  - Applies `GridSearchCV` for model tuning
  - Avoids data leakage by applying SMOTE inside the pipeline
  - Reported training accuracy: `0.8678`
  - Reported testing accuracy: `0.8533`

### 2. Logistic Regression

- [Logistic_Regression_Algorithm_Initial.ipynb](Logistic_Regression_Algorithm_Initial.ipynb)
  - Trains a Logistic Regression model on the heart dataset
  - Uses SMOTE and `StandardScaler`
  - Shows target distribution, confusion matrix, and learning curve
  - Reported training accuracy: `0.8695`
  - Reported testing accuracy: `0.8804`

- [Logistic_Regression_Algorithm_Optimized.ipynb](Logistic_Regression_Algorithm_Optimized.ipynb)
  - Uses `GridSearchCV` to tune Logistic Regression
  - Balances the training data with SMOTE
  - Includes accuracy plots and learning curve analysis
  - Best cross-validation accuracy: `0.8533742331288343`
  - Reported training accuracy: `0.8559`
  - Reported testing accuracy: `0.8913`

### 3. Random Forest

- [Random_Forest_Algorithm_Initial_Version.ipynb](Random_Forest_Algorithm_Initial_Version.ipynb)
  - Builds a Random Forest classifier for heart disease prediction
  - Uses preprocessing, feature encoding, and model evaluation
  - Reported test accuracy: `0.8532608695652174`

- [Random_Forest_Algorithm_Optimized.ipynb](Random_Forest_Algorithm_Optimized.ipynb)
  - Applies SMOTE to address class imbalance
  - Uses `GridSearchCV` / randomized search style tuning with a Random Forest pipeline
  - Includes confusion matrix and performance visualization
  - Reported training accuracy: `0.8991825613079019`
  - Reported test accuracy: `0.8641304347826086`

## Project Workflow

1. Load `heart.csv`
2. Clean and encode categorical variables
3. Split the dataset into train and test sets
4. Balance the training set using SMOTE
5. Train one of the classifiers
6. Tune hyperparameters in the optimized notebooks
7. Evaluate the final model using accuracy, classification report, and confusion matrix

## Requirements

The notebooks use the following Python packages:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `imbalanced-learn`

If you are running the notebooks locally, install the dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

## How To Run

1. Open the project folder in VS Code or Jupyter.
2. Open any of the `.ipynb` notebooks.
3. Run the cells from top to bottom.
4. Compare the initial and optimized versions to see the effect of preprocessing and tuning.

## Notes

- The optimized notebooks are designed to improve generalization and reduce bias from class imbalance.
- The Random Forest optimized notebook works with a slightly more complete preprocessing and tuning workflow than the initial version.
- Notebook filenames keep the original project naming, including the `Gussian` spelling.
