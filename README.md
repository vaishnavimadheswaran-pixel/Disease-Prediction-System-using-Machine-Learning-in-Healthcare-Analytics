# Disease Prediction System using Machine Learning in Healthcare Analytics

## 1. Project Overview

The **Disease Prediction System** is a machine learning project designed to predict the probable disease of a patient using medical information and symptoms.

The system analyzes patient data such as age, gender, blood pressure, sugar level, cholesterol, medical history, and symptoms. Different machine learning classification algorithms are trained and evaluated to identify the best-performing model.

This project is developed for **educational and healthcare analytics purposes**.

> **Disclaimer:** This system is an academic demonstration and is not a medical diagnostic tool. It must not replace professional medical advice or diagnosis.

---

## 2. Objectives

- Predict probable diseases from patient medical data.
- Analyze symptoms using machine learning.
- Compare different classification algorithms.
- Assist with early-risk analysis in an educational setting.
- Provide an interactive prediction interface.
- Display model performance using evaluation metrics.
- Visualize healthcare data using graphs.

---

## 3. Diseases Covered

The demonstration dataset contains the following disease classes:

- Diabetes
- Heart Disease
- Flu
- Gastritis
- Allergy
- Arthritis

---

## 4. Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Plotly
- Joblib

### Development Tool

- Visual Studio Code

### Dataset

- CSV format
- `disease_data.csv`

---

## 5. Machine Learning Algorithms

The project compares the following classification algorithms:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Naive Bayes
5. Support Vector Machine (SVM)

The models are trained using an **80% training and 20% testing split**.

The best-performing model is selected based on the evaluation results.

---

## 6. Dataset Features

The dataset contains patient information such as:

| Feature | Description |
|---|---|
| Patient_ID | Unique patient identification number |
| Age | Patient age |
| Gender | Patient gender |
| Blood_Pressure | Blood pressure level |
| Sugar_Level | Blood glucose level |
| Cholesterol | Cholesterol level |
| Medical_History | Previous medical history |
| Fever | Presence of fever |
| Cough | Presence of cough |
| Fatigue | Presence of fatigue |
| Headache | Presence of headache |
| Chest_Pain | Presence of chest pain |
| Shortness_of_Breath | Breathing difficulty |
| Nausea | Presence of nausea |
| Vomiting | Presence of vomiting |
| Joint_Pain | Presence of joint pain |
| Skin_Rash | Presence of skin rash |
| Disease | Target disease |

The symptom columns use:

- `1` = symptom present
- `0` = symptom absent

---

## 7. Project Architecture

```text
Patient Data
     |
     v
Data Collection
     |
     v
Data Preprocessing
     |
     v
Feature Engineering
     |
     v
Train/Test Split
     |
     v
Machine Learning Models
     |
     +----------------------+
     |          |           |
     v          v           v
Logistic    Decision     Random
Regression    Tree        Forest
     |          |           |
     +----------+-----------+
                |
                v
        Model Evaluation
                |
                v
          Best Model
                |
                v
       Disease Prediction
                |
                v
       Prediction Result
```

---

## 8. Project Modules

### 1. Data Input Module

Collects patient information including:

- Age
- Gender
- Blood pressure
- Sugar level
- Cholesterol
- Medical history
- Symptoms

### 2. Data Processing Module

Performs:

- Missing-value handling
- Categorical encoding
- Numerical scaling
- Data cleaning

### 3. Feature Engineering Module

Converts patient symptoms and medical information into machine-readable features.

### 4. Model Training Module

Trains multiple classification algorithms using the patient dataset.

### 5. Prediction Module

Accepts new patient information and predicts the most probable disease class.

### 6. Visualization Module

Displays:

- Disease distribution
- Age distribution
- Sugar-level analysis
- Model comparison
- Prediction probabilities

---

## 9. Model Evaluation

The following metrics are used:

### Accuracy

Measures the percentage of correctly classified records.

### Precision

Measures how many predicted positive results are correct.

### Recall

Measures how many actual positive cases are correctly identified.

### F1-Score

Provides a balance between precision and recall.

### Confusion Matrix

Shows the relationship between actual and predicted disease classes.

---

## 10. Installation

Clone or download the project and open it in Visual Studio Code.

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## 11. Dataset Setup

Make sure the project contains:

```text
data/
└── disease_data.csv
```

The dataset should be available before training the model.

---

## 12. Train the Model

Run:

```bash
python train_model.py
```

The program will:

1. Load the dataset.
2. Check the data.
3. Separate features and target.
4. Preprocess numerical and categorical data.
5. Split the data into training and testing sets.
6. Train five classification models.
7. Calculate accuracy, precision, recall, and F1-score.
8. Compare the models.
9. Select the best model.
10. Save the trained model.

The trained model will be saved in:

```text
models/disease_model.pkl
```

Model information will be saved in:

```text
models/model_info.pkl
```

---

## 13. Run the Application

After successful model training, run:

```bash
streamlit run app.py
```

The application will open in your browser.

Usually, Streamlit provides a local address such as:

```text
http://localhost:8501
```

---

## 14. Application Features

### Disease Prediction

The user can enter:

- Age
- Gender
- Blood pressure
- Sugar level
- Cholesterol
- Medical history
- Symptoms

The system then displays the predicted disease class.

### Dashboard

The dashboard displays:

- Total patient records
- Number of disease classes
- Disease distribution
- Age analysis
- Sugar-level analysis
- Patient data

### Model Performance

The application displays:

- Accuracy
- Precision
- Recall
- F1-score
- Model comparison

### Prediction Probability

When supported by the trained classifier, the application displays the model's probability distribution across disease classes.

---

## 15. Expected Output

Example:

```text
Patient Information

Age: 45
Gender: Male
Blood Pressure: 140
Sugar Level: 160
Cholesterol: 220

Symptoms:
Chest Pain
Dizziness
Shortness of Breath

Predicted Disease:
Heart Disease
```

The application also displays a probability chart for the model's predictions.

---

## 16. Advantages

- Easy to use.
- Fast prediction.
- Uses multiple machine learning algorithms.
- Provides model performance metrics.
- Provides graphical data analysis.
- Helps demonstrate healthcare analytics concepts.
- Can be extended with larger datasets.

---

## 17. Limitations

- Prediction quality depends on the dataset.
- The demonstration dataset is not clinical data.
- The system only predicts disease classes represented in the training dataset.
- Machine learning predictions may contain errors.
- The system is not clinically validated.
- It cannot replace doctors or healthcare professionals.

---

## 18. Future Enhancements

Future versions can include:

- Real clinical datasets with appropriate approvals.
- Larger and more diverse datasets.
- Deep learning models.
- Real-time health monitoring.
- Wearable-device integration.
- Mobile application.
- Cloud deployment.
- Explainable AI.
- Doctor/administrator dashboards.
- Secure patient authentication.
- Electronic health record integration.

---

## 19. Project Structure

```text
Disease Prediction System using Machine Learning in Healthcare Analytics/
│
├── app.py
├── generate_dataset.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── data/
│   └── disease_data.csv
│
└── models/
    ├── disease_model.pkl
    └── model_info.pkl
```

---

## 20. Quick Start

Run these commands in order:

```bash
pip install -r requirements.txt
```

```bash
python train_model.py
```

```bash
streamlit run app.py
```

---

## 21. Conclusion

The **Disease Prediction System using Machine Learning in Healthcare Analytics** demonstrates how machine learning classification techniques can be applied to patient symptoms and medical parameters for disease-class prediction.

The project combines data preprocessing, machine learning, model evaluation, visualization, and an interactive Streamlit interface into one healthcare analytics application.

It provides a strong academic demonstration of how data-driven systems can support healthcare analytics while highlighting the importance of professional medical judgment.

---

## 22. Medical Disclaimer

This project is intended strictly for **educational and academic purposes**.

The predictions generated by this application are not medical diagnoses and should not be used to make decisions about treatment, medication, emergency care, or other medical interventions.

Always consult a qualified healthcare professional for medical evaluation and diagnosis.