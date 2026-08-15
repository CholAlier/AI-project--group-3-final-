# Phoenix AI — Employee Attrition Prediction Platform

**Ashesi University | AI Group 3 Project**

## Project Overview

Phoenix AI is a machine learning platform that predicts employee attrition risk. It uses employee data to identify employees who may be at risk of leaving an organization, helping HR teams make proactive, data-driven retention decisions.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Logistic Regression
* Random Forest
* Streamlit
* Plotly
* Category Encoders

Project Structure

Employee Attrition Rate.ipynb--------Data analysis, preprocessing, model training, and evaluation.


phoenix_ml.py-----Machine learning pipeline for training and prediction.


app.py--------Streamlit application for employee attrition predictions.


HR-Employee-Attrition.csv-----IBM HR Analytics employee dataset used for training and testing.


requirements.txt-----------Required Python libraries for running the project.


## Machine Learning Models

### Logistic Regression

Used as an interpretable classification model for predicting whether an employee is likely to leave.

### Random Forest

An ensemble classification model that captures more complex relationships in the employee data.

### Combined Prediction

The system combines the predictions from both models to produce the final attrition risk prediction.

## Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* ROC-AUC
* Confusion Matrix

The dataset is divided into:

* **70% Training**
* **15% Validation**
* **15% Testing**

## Dataset

The project uses the **IBM HR Analytics Employee Attrition & Performance Dataset**.

* **Records:** 1,470 employees
* **Target variable:** Attrition
* **Attrition rate:** Approximately 16.1%

The dataset is preprocessed before model training by removing constant, unique, and redundant variables.

## Application Features

The Streamlit application provides:

* Individual employee attrition prediction
* Attrition risk score
* Comparison of Logistic Regression and Random Forest predictions
* Batch prediction using CSV files
* Workforce analytics and visualizations
* Feature importance analysis
* Basic fairness and ethics analysis
* Prediction reports and results export


### 2. Run the Application


streamlit run app.py


## Project Goal

The goal of Phoenix AI is to help organizations move from **reactive** employee management to **proactive** employee retention by identifying potential attrition risks early and supporting data-driven HR decisions.

## Business Impact

Phoenix AI can help organizations:

* Identify employees at risk of leaving
* Understand factors associated with employee attrition
* Support employee retention strategies
* Reduce potential recruitment and replacement costs
* Make better data-driven HR decisions
