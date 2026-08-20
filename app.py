import streamlit as st

import pandas as pd

import numpy as np

import joblib

import plotly.express as px

from pathlib import Path


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(

    page_title="Disease Prediction System",

    page_icon="🩺",

    layout="wide"
)


# ==========================================
# LOAD MODEL
# ==========================================

MODEL_PATH = Path(
    "models/disease_model.pkl"
)

INFO_PATH = Path(
    "models/model_info.pkl"
)

DATA_PATH = Path(
    "data/disease_data.csv"
)


if not MODEL_PATH.exists():

    st.error(
        "Model not found. Please run train_model.py first."
    )

    st.stop()


model = joblib.load(
    MODEL_PATH
)

model_info = joblib.load(
    INFO_PATH
)

df = pd.read_csv(
    DATA_PATH
)


# ==========================================
# TITLE
# ==========================================

st.title(
    "🩺 Disease Prediction System"
)

st.subheader(
    "Healthcare Analytics using Machine Learning"
)

st.warning(
    "This application is an academic demonstration. "
    "It is not a medical diagnostic tool and must not "
    "replace professional medical advice."
)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title(
    "Navigation"
)

page = st.sidebar.radio(

    "Select Module",

    [
        "Disease Prediction",
        "Dashboard",
        "Model Performance",
        "About Project"
    ]
)


# ==========================================
# SYMPTOMS
# ==========================================

symptoms = [

    "fever",
    "cough",
    "fatigue",
    "headache",
    "sore_throat",
    "body_pain",
    "shortness_of_breath",
    "chest_pain",
    "palpitations",
    "dizziness",
    "nausea",
    "vomiting",
    "abdominal_pain",
    "diarrhea",
    "frequent_urination",
    "excessive_thirst",
    "blurred_vision",
    "joint_pain",
    "skin_rash",
    "itching"
]


# ==========================================
# DISEASE PREDICTION
# ==========================================

if page == "Disease Prediction":

    st.header(
        "Patient Information"
    )

    col1, col2 = st.columns(2)


    # ======================================
    # PATIENT INFORMATION
    # ======================================

    with col1:

        age = st.number_input(

            "Age",

            min_value=1,

            max_value=120,

            value=30
        )


        gender = st.selectbox(

            "Gender",

            [
                "Male",
                "Female"
            ]
        )


        blood_pressure = st.number_input(

            "Blood Pressure",

            min_value=50,

            max_value=250,

            value=120
        )


        sugar_level = st.number_input(

            "Sugar Level",

            min_value=40,

            max_value=500,

            value=100
        )


        cholesterol = st.number_input(

            "Cholesterol",

            min_value=50,

            max_value=500,

            value=180
        )


        medical_history = st.selectbox(

            "Medical History",

            [
                "None",
                "Mild",
                "Moderate",
                "Strong"
            ]
        )


    # ======================================
    # SYMPTOMS
    # ======================================

    with col2:

        st.subheader(
            "Select Patient Symptoms"
        )

        selected_symptoms = st.multiselect(

            "Symptoms",

            symptoms
        )


    st.write("")


    # ======================================
    # PREDICT BUTTON
    # ======================================

    if st.button(

        "🔍 Predict Disease",

        type="primary"

    ):

        patient = {

            "Age":
                age,

            "Gender":
                gender,

            "Blood_Pressure":
                blood_pressure,

            "Sugar_Level":
                sugar_level,

            "Cholesterol":
                cholesterol,

            "Medical_History":
                medical_history
        }


        # Add symptoms

        for symptom in symptoms:

            if symptom in selected_symptoms:

                patient[symptom] = 1

            else:

                patient[symptom] = 0


        # Convert to DataFrame

        input_data = pd.DataFrame(
            [patient]
        )


        # Make sure feature order matches training

        input_data = input_data[
            model_info["features"]
        ]


        # ==================================
        # PREDICTION
        # ==================================

        prediction = model.predict(
            input_data
        )[0]


        st.success(
            f"Predicted Disease: {prediction}"
        )


        # ==================================
        # PROBABILITY
        # ==================================

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                input_data
            )[0]


            probability_df = pd.DataFrame({

                "Disease":
                    model.classes_,

                "Probability":
                    probabilities

            })


            probability_df = probability_df.sort_values(

                "Probability",

                ascending=False
            )


            st.subheader(
                "Prediction Probability"
            )


            fig = px.bar(

                probability_df,

                x="Disease",

                y="Probability",

                text_auto=".2f",

                title="Disease Prediction Probability"
            )


            fig.update_yaxes(
                range=[0, 1]
            )


            st.plotly_chart(

                fig,

                use_container_width=True
            )


            highest_probability = (
                probability_df.iloc[0]
            )


            st.info(

                f"Highest model probability: "
                f"{highest_probability['Probability']:.2%}"
            )


# ==========================================
# DASHBOARD
# ==========================================

elif page == "Dashboard":

    st.header(
        "📊 Healthcare Analytics Dashboard"
    )


    # ======================================
    # METRICS
    # ======================================

    col1, col2, col3 = st.columns(3)


    col1.metric(

        "Total Patients",

        len(df)
    )


    col2.metric(

        "Disease Classes",

        df["Disease"].nunique()
    )


    col3.metric(

        "Features",

        len(df.columns) - 1
    )


    # ======================================
    # DISEASE DISTRIBUTION
    # ======================================

    st.subheader(
        "Disease Distribution"
    )


    disease_count = (

        df["Disease"]

        .value_counts()

        .reset_index()
    )


    disease_count.columns = [

        "Disease",

        "Count"
    ]


    fig = px.bar(

        disease_count,

        x="Disease",

        y="Count",

        text="Count",

        title="Number of Patients by Disease"
    )


    st.plotly_chart(

        fig,

        use_container_width=True
    )


    # ======================================
    # AGE ANALYSIS
    # ======================================

    col1, col2 = st.columns(2)


    with col1:

        fig_age = px.histogram(

            df,

            x="Age",

            color="Disease",

            title="Age Distribution"
        )


        st.plotly_chart(

            fig_age,

            use_container_width=True
        )


    # ======================================
    # SUGAR ANALYSIS
    # ======================================

    with col2:

        fig_sugar = px.box(

            df,

            x="Disease",

            y="Sugar_Level",

            title="Sugar Level by Disease"
        )


        st.plotly_chart(

            fig_sugar,

            use_container_width=True
        )


    # ======================================
    # DATA TABLE
    # ======================================

    st.subheader(
        "Patient Dataset"
    )


    st.dataframe(

        df.head(50),

        use_container_width=True
    )


# ==========================================
# MODEL PERFORMANCE
# ==========================================

elif page == "Model Performance":

    st.header(
        "🤖 Machine Learning Model Performance"
    )


    results = pd.DataFrame(
        model_info["results"]
    ).T


    results = results.reset_index()


    results = results.rename(

        columns={
            "index": "Model"
        }
    )


    st.dataframe(

        results,

        use_container_width=True
    )


    # ======================================
    # MODEL COMPARISON
    # ======================================

    fig = px.bar(

        results,

        x="Model",

        y=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],

        barmode="group",

        title="Model Performance Comparison"
    )


    fig.update_yaxes(
        range=[0, 1]
    )


    st.plotly_chart(

        fig,

        use_container_width=True
    )


    st.success(

        f"Best Model: "
        f"{model_info['best_model']}"
    )


# ==========================================
# ABOUT PROJECT
# ==========================================

elif page == "About Project":

    st.header(
        "📚 About the Project"
    )


    st.markdown("""

## Disease Prediction System using Machine Learning

### Objective

The system predicts a probable disease class from patient symptoms and medical parameters.

### Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Matplotlib
- Seaborn

### Machine Learning Algorithms

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Naive Bayes
5. Support Vector Machine

### Methodology

1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis
4. Feature Engineering
5. Model Selection
6. Model Training
7. Model Evaluation
8. Disease Prediction

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### Limitations

The system depends on the quality and representativeness of its training data. It is not a replacement for medical professionals.

""")