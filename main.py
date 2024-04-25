import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to train models and create ensemble
def train_models_and_ensemble(X_train, Y_train):
    models = [
        LogisticRegression(),
        DecisionTreeClassifier(),
        RandomForestClassifier()
    ]
    accuracies = []
    # Train each model on the training data
    for model in models:
        model.fit(X_train, Y_train)
        # Calculate accuracy on training data
        train_accuracy = accuracy_score(Y_train, model.predict(X_train))
        # Calculate accuracy on test data
        test_accuracy = accuracy_score(Y_test, model.predict(X_test))
        accuracies.append((train_accuracy, test_accuracy))
    return models, accuracies

# Function to make predictions using ensemble
def predict_ensemble(models, X_test):
    # Make predictions using each model
    predictions = [model.predict(X_test) for model in models]
    # Combine the predictions using a voting mechanism
    ensemble_predictions = sum(predictions) / len(models)
    # Convert probabilities to class labels
    ensemble_predictions = [1 if pred >= 0.5 else 0 for pred in ensemble_predictions]
    return ensemble_predictions

# Function to predict on a new instance using ensemble
def predict_new_instance_with_models(models, input_data):
    ensemble_prediction = predict_ensemble(models, input_data)
    if ensemble_prediction == 0:
        return 'Congratulations! Based on our analysis, it seems like the heart of the individual is as strong as an ox! No signs of heart disease detected!'
    else:
        return 'Uh-oh! Our analysis indicates that there might be some irregularities in the individual\'s heart health. We recommend consulting a healthcare professional for further evaluation.'

# Load the dataset
heart_data = pd.read_csv('data.csv')

# Split the data into features and target
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# Train models and create ensemble
models, accuracies = train_models_and_ensemble(X_train, Y_train)

# Streamlit UI
st.set_page_config(page_title="Heart Disease Predictor", page_icon="💝")

st.markdown(
    ''
    '<div style="position: fixed; width: 45.85%; top: 25; background-color: #ffffff; z-index: 1; text-align: center;"><h2>🩺 HeartGuard: Predicting Heart Health</h2></div>',
    unsafe_allow_html=True
)

# Display image
st.image('heart_image.jpg', use_column_width=True)

# User input section
st.header('Enter Patient Details')

# Center align the first element in a row
with st.expander("Age", expanded=True):
    age = st.slider('Age', min_value=20, max_value=90, value=50, help="Age of the patient")

# Create two columns for the remaining features
col1, col2 = st.columns(2)

with col1:
    with st.expander("Sex", expanded=True):
        sex = st.radio('Sex', ['Male', 'Female'], help="Sex of the patient")

    with st.expander("Chest Pain Type", expanded=True):
        cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'], help="Type of chest pain experienced by the patient, For info on Angina : https://www.mayoclinic.org/diseases-conditions/angina/symptoms-causes/syc-20369373#:~:text=Angina%20(an%2DJIE%2Dnuh,or%20pain%20in%20the%20chest.")

    with st.expander("Resting Blood Pressure", expanded=True):
        trestbps = st.slider('Resting Blood Pressure (mm Hg)', min_value=90, max_value=200, value=120, help="Resting blood pressure in mm Hg: https://www.cdc.gov/bloodpressure/about.htm#:~:text=Blood%20pressure%20is%20measured%20using,your%20heart%20rests%20between%20beats.")

    with st.expander("Serum Cholesterol", expanded=True):
        chol = st.slider('Serum Cholesterol (mg/dl)', min_value=100, max_value=400, value=200, help="Serum cholesterol level of the patient in mg/dl. Fo more info: https://www.medicalnewstoday.com/articles/321519#:~:text=A%20person's%20serum%20cholesterol%20level%20comprises%20the%20amount%20of%20high,conditions%20such%20as%20heart%20disease.")

    with st.expander("Fasting Blood Sugar", expanded=True):
        fbs = st.selectbox('Fasting Blood Sugar (> 120 mg/dl)', ['<= 120 mg/dl', '> 120 mg/dl'], help="Fasting blood sugar level, For more info : https://my.clevelandclinic.org/health/diagnostics/21952-fasting-blood-sugar")

    with st.expander("Resting Electrocardiographic Results", expanded=True):
        restecg = st.selectbox('Resting Electrocardiographic Results',
                               ['Normal', 'ST-T Wave Abnormality', 'Probable or Definite Left Ventricular Hypertrophy'], help="Resting electrocardiogram, For more info : https://www.ncbi.nlm.nih.gov/books/NBK367910/#:~:text=Resting%2012%2Dlead%20electrocardiography%20(ECG,hypertrophy%20and%20bundle%20branch%20blocks.")

with col2:
    with st.expander("Maximum Heart Rate Achieved", expanded=True):
        thalach = st.slider('Maximum Heart Rate Achieved (bpm)', min_value=60, max_value=220, value=150, help="Maximum heart rate, Fo more info : https://www.hopkinsmedicine.org/health/wellness-and-prevention/understanding-your-target-heart-rate#:~:text=The%20maximum%20rate%20is%20based,or%2085%20beats%20per%20minute.")

    with st.expander("Exercise Induced Angina", expanded=True):
        exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'], help="Whether exercise induced angina was observed in the patient, For more info: https://my.clevelandclinic.org/health/diseases/21489-angina")

    with st.expander("ST Depression Induced by Exercise", expanded=True):
        oldpeak = st.slider('ST Depression Induced by Exercise', min_value=0.0, max_value=6.0, value=2.0, help="ST depression induced by exercise relative to rest, For more info: https://pubmed.ncbi.nlm.nih.gov/10707755/#:~:text=Objective%3A%20ST%2Dsegment%20depression%20is,ST%2Dsegment%20depression%20remains%20unclear.")

    with st.expander("Slope of Peak Exercise ST Segment", expanded=True):
        slope = st.selectbox('Slope of Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'], help="More info: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1123032/#:~:text=The%20J%20point%20(the%20point,exercise%20therefore%20slopes%20sharply%20upwards.")

    with st.expander("Number of Major Vessels Colored by Flourosopy", expanded=True):
        ca = st.selectbox('Number of Major Vessels Colored by Flourosopy', [0, 1, 2, 3], help="Number of major vessels colored by fluoroscopy, More info: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4468223/")

    with st.expander("Thalassemia", expanded=True):
        thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'], help="Thalassemia category, For more info: https://www.mayoclinic.org/diseases-conditions/thalassemia/symptoms-causes/syc-20354995#:~:text=Thalassemia%20(thal%2Duh%2DSEE,you%20might%20not%20need%20treatment.")

# Convert sex to binary
sex = 1 if sex == 'Male' else 0

# Convert descriptive options to numerical values
cp_mapping = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3}
fbs_mapping = {'<= 120 mg/dl': 0, '> 120 mg/dl': 1}
restecg_mapping = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Probable or Definite Left Ventricular Hypertrophy': 2}
exang_mapping = {'No': 0, 'Yes': 1}
slope_mapping = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
thal_mapping = {'Normal': 0, 'Fixed Defect': 1, 'Reversible Defect': 2}

# Convert input into DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp_mapping[cp]],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs_mapping[fbs]],
    'restecg': [restecg_mapping[restecg]],
    'thalach': [thalach],
    'exang': [exang_mapping[exang]],
    'oldpeak': [oldpeak],
    'slope': [slope_mapping[slope]],
    'ca': [ca],
    'thal': [thal_mapping[thal]]
})

# Custom CSS
st.markdown(
    '''
    <style>
    .streamlit-expanderHeader {
        background-color: white;
        color: black; # Adjust this for expander header color
    }
    .streamlit-expanderContent {
        background-color: #F48D8D;
        color: black; # Expander content color
    }
    </style>
    ''',
    unsafe_allow_html=True
)

# Predict on new instance using ensemble
result = predict_new_instance_with_models(models, input_data)

# Center align the button and make it larger
# st.markdown("<h2 style='text-align: center;'><button style='background-color: #4CAF50; color: white; padding: 20px; text-align: center; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 10px; width: 100%; border: none;'>Predict</button></h2>", unsafe_allow_html=True)

# Show prediction result with a pop-up message
if st.button('Predict', help="Click to predict"):
    result = predict_new_instance_with_models(models, input_data)
    if 'Congratulations' in result:
        st.success(result)
    else:
        st.error(result)

import streamlit as st

# Explanation of model workings
st.markdown("""
    <div style="font-size: 14px; margin-top: 20px; margin-bottom: 20px;">
        <p><strong>How does our model work?</strong></p>
        <p>Our Heart Disease Prediction model is based on machine learning algorithms - Logistic Regression, Decision Tree, and Random Forest - trained on a dataset of various heart health indicators. When you input your details and click the 'Predict' button, the model evaluates the information provided and compares it with patterns it has learned from past data.</p>
        <p>If the model detects no signs of heart disease, it will provide a prediction indicating that the individual's heart health seems strong. On the other hand, if the model finds irregularities that might indicate heart issues, it will recommend consulting a healthcare professional for further evaluation.</p>
        <p>This model is designed to provide an initial assessment based on common heart health indicators. However, it's essential to note that the prediction should not be considered a substitute for professional medical advice.</p>
    </div>
""", unsafe_allow_html=True)

# Set custom CSS
st.markdown(
    """
    <style>
    body, .stApp {
        margin-top: -60px;
        background-image: linear-gradient(to bottom right, #ffb3c2, #ffe6eb);
    }
    .footer {
        position : static;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f7a3cd !important;
        padding: 10px;
        text-align: center;
        color: #fff !important;
        font-size: 14px;
    }
    .streamlit-expanderHeader {
        background-color: white;
        color: black;
    }
    .streamlit-expanderContent {
        background-color: #F48D8D;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Footer content
st.markdown('<div class="footer">Developed by <a href="https://www.linkedin.com/in/priyanshu-pandya/" style="color: white; text_decoration:none">Priyanshu Pandya</a></div>', unsafe_allow_html=True)
