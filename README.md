# Heart Disease Predictor 🩺

Welcome to HeartGuard, a web application for predicting heart disease risk based on various health indicators. This application leverages machine learning algorithms to analyze user-provided data and provide insights into heart health.
With a focus on early detection and prevention, it contributes to promoting heart health awareness and improving overall well-being.
Our Heart Disease Predictor model is based on machine learning algorithms - Logistic Regression, Decision Tree, and Random Forest - trained on a dataset of various heart health indicators. When you input your details and click the 'Predict' button, the model evaluates the information provided and compares it with patterns it has learned from past data.
If the model detects no signs of heart disease, it will provide a prediction indicating that the individual's heart health seems strong. On the other hand, if the model finds irregularities that might indicate heart issues, it will recommend consulting a healthcare professional for further evaluation.

## Features

- Predicts heart disease risk based on user input.
- Utilizes machine learning algorithms: Logistic Regression, Decision Tree, and Random Forest.
- Provides an ensemble prediction for optimal accuracy.
- Streamlit-based user interface for easy interaction.

## Data Resource
- The Heart Disease data used in this project was sourced from the kaggle.
- Link: `https://www.kaggle.com/datasets/priyanka841/heart-disease-prediction-uci`
- The dataset used in this project contains various health indicators related to heart disease, such as age, sex, chest pain type, resting blood pressure, serum cholesterol level, and more.
- These features serve as inputs to the machine learning models for predicting the likelihood of heart disease.

## Set-Up
To run the HeartGuard locally, follow these stepsg:

- Python 3.3 or higher installed on your system.
- Install the required Python Libraries Scikit-learn and Pandas using pip 
- Have Streamlit for Quick UI setup.
- Run the Streamlit application using streamlit run app.py.
- Access the application in your web browser at `http://localhost:8501`.

## Methodology
- **Data Collection**: The dataset containing various heart health indicators is loaded from data.csv.
- **Data Preprocessing**: The dataset is split into features and target variables. Categorical variables are encoded, and missing values are handled if any.
- **Model Training**: Three machine learning algorithms (Logistic Regression, Decision Tree, and Random Forest) are trained on the training data to predict heart disease risk.
- **Ensemble Method**: Predictions from all three models are combined using a voting mechanism to create an ensemble prediction for optimal accuracy.
- **User Interface**: Streamlit is used to create a user-friendly interface where users can input their details and get predictions about their heart health.
