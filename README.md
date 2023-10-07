# Credit Card Fraud Detection Project

## Introduction
This project aims to build a machine learning model that can detect fraudulent credit card transactions based on the features of the transactions. The data used for this project is from a [Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) that contains anonymized credit card transactions labeled as 1 or 0 ie. fraudulent or valid. The dataset is highly imbalanced, with only 0.2% of the transactions being fraudulent.
<img src="https://github.com/Farah-DeebaJ/CreditCardFraudDetection/blob/main/distribution.png">


## Data Preprocessing
The data preprocessing steps include:

- Sampling the data to balance the classes and reduce the computational cost
- Splitting the data into two sets: training and testing
- Scaling the features using standardization

## Model Building and Evaluation
The models used for this project are:

- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier
- Gradient Boosting Classifier
- AdaBoost Classifier
- XGBoost Classifier
- CatBoost Classifier

The models are trained on the training set and evaluated on the validation set using the following metrics:

- Precision: the proportion of predicted fraudulent transactions that are actually fraudulent
- Recall: the proportion of actual fraudulent transactions that are correctly predicted as fraudulent
- F1 score: the harmonic mean of precision and recall
- ROC curve: a plot of the true positive rate versus the false positive rate at different threshold levels
- AUC: the area under the ROC curve, which measures the overall performance of the model

## Feature Importance Analysis
The feature importance analysis is done using SHAP (SHapley Additive exPlanations), a framework that computes and visualizes the feature contributions for each prediction. SHAP values can show how much each feature influences the prediction, and in which direction. The SHAP plots used for this project is Summary plot which depicts a global overview of the impact of all features on the model output.

Hello, this is Bing. I can help you write a readme file for your credit card fraud detection project. Here is a possible template that you can use:

## Visualization

I created two graphs using PowerBI to compare the results of different models:

- **Cross Validation Accuracy**: This graph shows the mean cross-validation accuracy of each model on the training and validation set.
<img src="https://github.com/Farah-DeebaJ/CreditCardFraudDetection/blob/main/Accuracy.png">
- **Performance Metrics**: This graph shows the training accuracy, testing accuracy, training precision, testing precision, training recall and testing recall of each model.
<img src="https://github.com/Farah-DeebaJ/CreditCardFraudDetection/blob/main/Performance.png">


## Conclusion and Future Work
The conclusion of this project is as follows:
- **XGBoost classifier** and **CatBoost classifier** is the best model for credit card fraud detection among the models that I have tried. It has the highest cross-validation accuracy of **0.96** and **0.95** respectively,the highest testing accuracy of 0.94. It also has the highest testing precision of **0.95** and **0.96** respectively and testing recall of **0.92** and **0.91** respectively, which means that it can correctly identify most of the fraudulent transactions and minimize the false positives and false negatives. 
- The **SHAP explainer** shows that the most important features for the model predictions are **V4**, **V10**, **V14** and **V12**, which are some of the anonymized features in the dataset. These features have the highest absolute SHAP values, which indicate how much they contribute to the model output.

The future work suggests some possible improvements and extensions for the project, such as:
- Testing the models on unseen data or real-world data
- Tuning the hyperparameters of the models to optimize their performance
- Exploring other methods for dealing with imbalanced data, such as SMOTE or cost-sensitive learning.
- Incorporating other features or external data sources that might be relevant for fraud detection.
