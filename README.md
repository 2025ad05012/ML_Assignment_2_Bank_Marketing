
# Bank Marketing Classification

## M.Tech AI/ML – Machine Learning Assignment 2

---

## 1. Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models for predicting whether a customer will subscribe to a term deposit based on bank marketing campaign information.

The project covers the complete machine learning workflow including data preprocessing, model training, evaluation, model persistence, and deployment through an interactive Streamlit application.

---

## 2. Dataset Description

The project uses a public Bank Marketing classification dataset.

The original dataset contains:

- 41,188 instances
- 21 columns
- 20 input features
- 1 target variable: `y`

The target variable represents whether the customer subscribed to a term deposit:

- `no`
- `yes`

### Target Distribution

| Target | Count | Percentage |
|---|---:|---:|
| no | 36,548 | 88.73% |
| yes | 4,640 | 11.27% |

### Duplicate Analysis

The original dataset contained 12 duplicate rows.

After removing duplicates:

- Final dataset size: 41,176 rows
- Duplicate rows remaining: 0

### Feature Types

There are:

- 10 numerical features
- 10 categorical features

### Numerical Features

- age
- duration
- campaign
- pdays
- previous
- emp.var.rate
- cons.price.idx
- cons.conf.idx
- euribor3m
- nr.employed

### Categorical Features

- job
- marital
- education
- default
- housing
- loan
- contact
- month
- day_of_week
- poutcome

---

## 3. Data Preprocessing

The dataset was checked for missing values and duplicate records.

The original dataset contained no missing values.

The 12 duplicate rows were removed before model development.

A preprocessing pipeline was created for the numerical and categorical features.

The target variable `y` was converted into binary values:

- `no` → 0
- `yes` → 1

After preprocessing:

- Training samples: 32,940
- Test samples: 8,236
- Processed features: 63

The processed training and test datasets were checked for NaN and infinite values. No NaN or infinite values were found.

---

## 4. Train-Test Split

After duplicate removal, the dataset was divided into training and test sets using a stratified split.

| Dataset | Samples |
|---|---:|
| Training Set | 32,940 |
| Test Set | 8,236 |

The target distribution was preserved.

### Training Target Distribution

| Class | Proportion |
|---|---:|
| No | 88.73% |
| Yes | 11.27% |

### Test Target Distribution

| Class | Proportion |
|---|---:|
| No | 88.73% |
| Yes | 11.27% |

---

## 5. Machine Learning Models Used

The following five classification models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier (Ensemble)

---

## 6. Evaluation Metrics

Each model was evaluated using the following six metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## 7. Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9095 | 0.9390 | 0.6522 | 0.4224 | 0.5128 | 0.4788 |
| Decision Tree | 0.8902 | 0.7246 | 0.5130 | 0.5108 | 0.5119 | 0.4500 |
| kNN | 0.9014 | 0.8741 | 0.5858 | 0.4267 | 0.4938 | 0.4474 |
| Gaussian Naive Bayes | 0.8227 | 0.8404 | 0.3506 | 0.6724 | 0.4609 | 0.3950 |
| Random Forest (Ensemble) | **0.9131** | **0.9443** | **0.6672** | 0.4558 | **0.5416** | **0.5065** |

---

## 8. Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Logistic Regression achieved strong overall performance with 90.95% Accuracy and 93.90% AUC. It achieved high Precision of 65.22% and an F1 Score of 51.28%, providing a good balance for the imbalanced classification problem. |
| Decision Tree | Decision Tree achieved 89.02% Accuracy. Its Recall of 51.08% was higher than Logistic Regression, but its AUC of 72.46% and Precision of 51.30% were lower than the stronger-performing models. |
| kNN | kNN achieved 90.14% Accuracy and 87.41% AUC. It produced moderate Precision of 58.58%, but its F1 Score of 49.38% and MCC of 44.74% were lower than Logistic Regression and Random Forest. |
| Gaussian Naive Bayes | Gaussian Naive Bayes achieved the highest Recall of 67.24%, identifying the largest proportion of positive cases. However, its Precision was only 35.06%, with Accuracy of 82.27%, indicating a higher number of false-positive predictions. |
| Random Forest (Ensemble) | Random Forest achieved the best overall performance, with the highest Accuracy of 91.31%, AUC of 94.43%, Precision of 66.72%, F1 Score of 54.16%, and MCC of 50.65%. |

### Overall Winner for the Dataset

**Random Forest (Ensemble)** is the overall winner for this dataset because it achieved the highest Accuracy, AUC, Precision, F1 Score, and MCC among the evaluated models.

Gaussian Naive Bayes achieved the highest Recall at 67.24%, but its lower Precision, Accuracy, F1 Score, and MCC resulted in weaker overall performance.

---

## 9. Confusion Matrix Results

### Logistic Regression

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 7,099 | 209 |
| Actual Yes | 536 | 392 |

### Decision Tree

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 6,858 | 450 |
| Actual Yes | 454 | 474 |

### kNN

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 7,028 | 280 |
| Actual Yes | 532 | 396 |

### Gaussian Naive Bayes

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 6,152 | 1,156 |
| Actual Yes | 304 | 624 |

### Random Forest

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 7,097 | 211 |
| Actual Yes | 505 | 423 |

---

## 10. Model Persistence

All trained models were saved using Joblib.

The following files are included:

- `decision_tree.pkl`
- `knn.pkl`
- `logistic_regression.pkl`
- `naive_bayes.pkl`
- `preprocessor.pkl`
- `random_forest.pkl`

The models were saved using compression to reduce file size.

All saved models and the preprocessing pipeline were successfully reloaded.

Prediction consistency was also verified. The predictions generated by the reloaded models were identical to the predictions generated before saving for all five models.

---

## 11. Test Dataset

The held-out test dataset used for the Streamlit demonstration is provided as:

`test_data.csv`

Test dataset dimensions:

- Rows: 8,236
- Columns: 21
- Input features: 20
- Target column: `y`

Verification results:

- Missing values: 0
- Duplicate rows: 0
- No class: 7,308
- Yes class: 928

---

## 12. Streamlit Application

An interactive Streamlit web application was developed for demonstrating the classification models.

The application provides the following features:

1. CSV test-data upload
2. Model selection dropdown
3. Prediction summary
4. Accuracy
5. AUC
6. Precision
7. Recall
8. F1 Score
9. MCC
10. Confusion matrix
11. Classification report

The application was tested successfully with all five implemented models.

The evaluation results displayed in Streamlit were verified against the notebook results.

### Live Application

**Streamlit Application:**  
https://mlassignment2bankmarketing-4kywwkcyaxwngwwtncoe9u9.streamlit.app/

---

## 13. GitHub Repository Link

The complete source code, trained model files, test dataset, requirements file, and README are available in the following GitHub repository:

**GitHub Repository:**  
https://github.com/2025ad05012/ML_Assignment_2_Bank_Marketing

---

## 14. Project Structure

```text
ML_Assignment_2_Bank_Marketing/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
└── model/
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── logistic_regression.pkl
    ├── naive_bayes.pkl
    ├── preprocessor.pkl
    └── random_forest.pkl
```
## 15. Requirements

The project uses the following Python libraries:

- Streamlit 1.61.1
- scikit-learn 1.9.0
- pandas
- NumPy
- Joblib
- Matplotlib
- Seaborn

The complete list is available in `requirements.txt`.

---

## 16. Conclusion

In this assignment, five classification models were implemented and compared using the same Bank Marketing dataset.

Random Forest performed best overall. It achieved the highest Accuracy, AUC, Precision, F1 Score and MCC among the five models.

Gaussian Naive Bayes achieved the highest Recall, but its Precision and Accuracy were lower than the other stronger-performing models.

The models were saved and tested again after loading them from the saved files. The Streamlit application was also tested using the held-out test dataset and the results matched the notebook results.
