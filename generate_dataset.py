import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

diseases = [
    "Diabetes",
    "Heart Disease",
    "Flu",
    "Gastritis",
    "Allergy",
    "Arthritis"
]

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

data = []

for patient_id in range(1, 2001):

    disease = random.choice(diseases)

    age = random.randint(18, 80)

    gender = random.choice([
        "Male",
        "Female"
    ])

    blood_pressure = random.randint(
        90,
        180
    )

    sugar_level = random.randint(
        60,
        250
    )

    cholesterol = random.randint(
        120,
        300
    )

    medical_history = random.choice([
        "None",
        "Mild",
        "Moderate",
        "Strong"
    ])

    row = {
        "Patient_ID": patient_id,
        "Age": age,
        "Gender": gender,
        "Blood_Pressure": blood_pressure,
        "Sugar_Level": sugar_level,
        "Cholesterol": cholesterol,
        "Medical_History": medical_history
    }

    for symptom in symptoms:

        probability = 0.05

        if disease == "Diabetes":

            if symptom in [
                "frequent_urination",
                "excessive_thirst",
                "blurred_vision",
                "fatigue"
            ]:
                probability = 0.85

        elif disease == "Heart Disease":

            if symptom in [
                "chest_pain",
                "shortness_of_breath",
                "palpitations",
                "dizziness"
            ]:
                probability = 0.85

        elif disease == "Flu":

            if symptom in [
                "fever",
                "cough",
                "headache",
                "sore_throat",
                "body_pain"
            ]:
                probability = 0.85

        elif disease == "Gastritis":

            if symptom in [
                "nausea",
                "vomiting",
                "abdominal_pain",
                "diarrhea"
            ]:
                probability = 0.85

        elif disease == "Allergy":

            if symptom in [
                "skin_rash",
                "itching",
                "cough",
                "sore_throat"
            ]:
                probability = 0.85

        elif disease == "Arthritis":

            if symptom in [
                "joint_pain",
                "body_pain",
                "fatigue"
            ]:
                probability = 0.85

        row[symptom] = np.random.binomial(
            1,
            probability
        )

    # Add some disease-related medical patterns

    if disease == "Diabetes":
        row["Sugar_Level"] = random.randint(130, 240)

    if disease == "Heart Disease":
        row["Blood_Pressure"] = random.randint(120, 180)
        row["Cholesterol"] = random.randint(180, 300)

    row["Disease"] = disease

    data.append(row)


df = pd.DataFrame(data)

df.to_csv(
    "data/disease_data.csv",
    index=False
)

print("Dataset created successfully!")
print("Number of records:", len(df))
print("Saved as: data/disease_data.csv")