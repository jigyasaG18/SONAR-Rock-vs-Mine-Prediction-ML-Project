# SONAR Rock vs Mine Prediction Machine Learning Project

### Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model Description](#model-description)
4. [Methodology](#methodology)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Model Comparison](#model-comparison)
7. [Confusion Matrix](#confusion-matrix)
8. [Inference from Confusion Matrices](#inference-from-confusion-matrices)
9. [Code Organization](#code-organization)
10. [Prerequisites](#prerequisites)
11. [Running the Code](#running-the-code)
12. [License](#license)

## Overview

This project implements various machine learning techniques to develop a model capable of predicting whether an object is a rock or a mine based on SONAR return data. The dataset used for this classification problem consists of SONAR return data, and multiple classification models are applied and compared to determine which one performs best.

## Dataset

The SONAR Rock vs Mine Prediction dataset consists of features derived from SONAR return data. It includes 208 instances, each with 60 numerical features representing the energy within frequency bands. The target labels are 'M' (mines) and 'R' (rocks).

### Dataset Overview

| Feature          | Description                                            |
|------------------|--------------------------------------------------------|
| 60 numerical values | Features representing energy levels over frequency bands |
| 'M' / 'R'        | Class labels for mines and rocks                       |

## Model Description

This project employs several classification models, focusing on:

1. **Logistic Regression**: A simple yet effective model for binary classification.
2. **Support Vector Classifier (SVC)**: Known for its effectiveness in high-dimensional spaces.
3. **Decision Tree Classifier**: A non-parametric supervised learning method.
4. **Random Forest Classifier**: An ensemble method that combines multiple decision trees.

## Methodology

The overall methodology of the project can be summarized in the following steps:

1. **Data Loading**: Load the dataset into a pandas DataFrame.
2. **Data Visualization and Summary**: Explore the dataset using descriptive statistics and visualizations to understand patterns.
3. **Data Preprocessing**: Separate features and labels, and split the dataset into training and testing sets.
4. **Model Training**: Train various classification models using the training data.
5. **Model Evaluation**: Evaluate each model's performance using accuracy metrics and confusion matrices.

## Model Training and Evaluation

### Training

Each of the classifiers is trained on the training data. For example, here's how the Logistic Regression model is trained:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, Y_train)
```

### Evaluation

The accuracy results for the tested models are as follows:

| Model                        | Accuracy   |
|------------------------------|------------|
| Logistic Regression          | 76.19%     |
| Support Vector Classifier    | 80.95%     |
| Decision Tree Classifier     | 71.43%     |
| Random Forest Classifier     | 76.19%     |

## Model Comparison

In this project, each model's performance was compared based on accuracy scores after training on the same dataset.

### Model Comparison Code

```python
from sklearn.metrics import accuracy_score

models = {
    "Logistic Regression": LogisticRegression(),
    "Support Vector Classifier": SVC(),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "Random Forest Classifier": RandomForestClassifier()
}

accuracy_scores = {}

for name, model in models.items():
    classifier = model.fit(X_train, Y_train)
    pred = classifier.predict(X_test)
    accuracy_scores[name] = accuracy_score(Y_test, pred)

classification_model_df = pd.DataFrame(accuracy_scores.items(), columns=["Model", "Accuracy"])
print(classification_model_df)
```

### Performance Results

| Model                        | Accuracy   |
|------------------------------|------------|
| Logistic Regression          | 76.19%     |
| Support Vector Classifier    | 80.95%     |
| Decision Tree Classifier     | 71.43%     |
| Random Forest Classifier     | 76.19%     |

## Confusion Matrix

The confusion matrices for the models are as follows:

### Logistic Regression

```
Confusion Matrix:
[[9 2]
 [3 7]]
```

### Support Vector Classifier

```
Confusion Matrix:
[[10  1]
 [ 3  7]]
```

## Inference from Confusion Matrices

### Logistic Regression

The confusion matrix for the Logistic Regression model is:

```
[[9 2]
 [3 7]]
```

![image](https://github.com/user-attachments/assets/846931cb-8e4e-4b90-ac82-d1674ad82e3a)


- **True Positives (TP)**: 9 (Mines correctly predicted as mines)
- **True Negatives (TN)**: 7 (Rocks correctly predicted as rocks)
- **False Positives (FP)**: 2 (Rocks incorrectly predicted as mines)
- **False Negatives (FN)**: 3 (Mines incorrectly predicted as rocks)

**Interpretations from the Confusion Matrix:**

1. **Accuracy**: The model achieves an accuracy of approximately 76.19%, suggesting a reasonable level of classification performance.

2. **Precision**: The precision for the positive class (mines) is calculated as:

   Precision = {TP} / {TP + FP} = {9} / {9+2} which is approx 0.818 (or 81.8%)
   This indicates that among all instances predicted as mines, 81.8% were actually mines.

3. **Recall**: The recall for the positive class (mines) is calculated as:
   
   Recall = {TP} / {TP + FN} = {9} / {9+3} which is approx 0.75 (or 75%)
   This means that the model correctly identifies 75% of actual mines.

4. **F1 Score**: The F1 score can be computed as:
   
   F1 Score = {2 * {Precision} * {Recall}} / {{Precision} + {Recall}} = {2 * 0.818 * 0.75} / {0.818 + 0.75} which is approx 0.782
   This simplifies to F1 Score to approx 0.782 (or 78.2%).

5. **Class Imbalance Consideration**: The model misclassified 2 rocks as mines (false positives), suggesting a possible tendency to overpredict mines, which may need to be addressed.

### Support Vector Classifier

The confusion matrix for the Support Vector Classifier is:

```
[[10  1]
 [ 3  7]]
```

![image](https://github.com/user-attachments/assets/8ba7f599-fd4f-46f5-ada9-78e450fa73d7)


- **True Positives (TP)**: 10 (Mines correctly predicted as mines)
- **True Negatives (TN)**: 7 (Rocks correctly predicted as rocks)
- **False Positives (FP)**: 1 (Rock incorrectly predicted as a mine)
- **False Negatives (FN)**: 3 (Mines incorrectly predicted as rocks)

**Interpretations from the Confusion Matrix:**

1. **Accuracy**: The SVC achieves an accuracy of approximately 80.95%, indicating better performance compared to the Logistic Regression model.

2. **Precision**: The precision for the positive class (mines) is calculated as:
   
   Precision = {TP} / {TP + FP} = {10} / {10+1} which is approx 0.909 (or 90.9%)
   This shows that 90.9% of the instances predicted as mines were indeed mines.

3. **Recall**: The recall for the positive class (mines) is:
   
   Recall = {TP} / {TP + FN} = {10} / {10+3} which is approx 0.769 (or 76.9%)
   This indicates that the model accurately identifies 76.9% of actual mines.

4. **F1 Score**: The F1 score can be computed as:
   
   F1 Score = {2 * {Precision} * {Recall}} / {{Precision} + {Recall}} = {2 * 0.909 * 0.769} / {0.909 + 0.769} which is approx 0.833
   This simplifies to F1 Score to approx 0.833 (or 83.3%).

5. **Class Imbalance Consideration**: The SVC has only one false positive, which suggests that it has a very good precision and is quite limited in misclassifying rocks as mines.

## Code Organization

The code is organized into the following logical sections:

1. Data Importing and Preprocessing
2. Model Training and Evaluation
3. Model Comparison
4. Visualization and Prediction

## Prerequisites

To run the code successfully, ensure you have the following libraries installed:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install them using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Running the Code

To execute the project:

1. Make sure you have all the required libraries installed.
2. Open the Jupyter Notebook containing the code snippets provided in this README.
3. Run the cells sequentially to train the models and evaluate their performances.

## Streamlit Deployment

The project has been deployed as a web application using Streamlit. You can access the interactive application through the following link:

#SONAR Rock vs Mine Prediction Streamlit App[https://sonar-rock-vs-mine-prediction-ml-app.streamlit.app/]

This app allows users to input SONAR data and receive real-time predictions regarding whether the object is a rock or a mine. 

How to Use the Streamlit App:
Input Form: Enter the values for SONAR returns in the provided input fields.
Prediction: Click on the "Predict" button to receive the prediction on whether the input corresponds to a rock or a mine.
Results Display: The application will display the results clearly on the screen.

## License

This project is licensed under the MIT License.
