import pickle
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os
import google.generativeai as genai
from IPython.display import Markdown
import warnings
warnings.filterwarnings('ignore')

# Set your API key
os.environ['GOOGLE_API_KEY'] = "YOUR_GOOGLE_API_KEY"

# Configure the API with the key
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Initialize the Gemini model (gemini-pro)
model = genai.GenerativeModel('gemini-1.5-pro')   # We will use the gemini-1.5-pro model as our LLM


# Load the saved model and preprocessing tools
rf_model = joblib.load("random_forest_model.pkl")  # Load trained model
scaler = joblib.load("scaler.pkl")  # Load saved standard scaler
ordinal_encoder = joblib.load("ordinal_encoder.pkl")  # Load saved ordinal encoder

# Define all features used in model training
all_features = [
    "Age", "self_employed", "family_history", "work_interfere", "no_employees",
    "remote_work", "tech_company", "benefits", "care_options", "wellness_program",
    "seek_help", "anonymity", "leave", "mental_health_consequence", "phys_health_consequence",
    "coworkers", "supervisor", "mental_health_interview", "phys_health_interview",
    "mental_vs_physical", "obs_consequence"
]


# Define selected features used for final prediction
selected_features = [
    "family_history", "work_interfere", "no_employees", "care_options", "leave",
    "benefits", "coworkers", "mental_health_consequence", "phys_health_interview",
    "mental_vs_physical", "supervisor", "seek_help", "wellness_program"
]

# Define binary and ordinal mappings
binary_mappings = {'No': 0, 'Yes': 1}

ordinal_mappings = {
    'work_interfere': ['Never', 'Rarely', 'Sometimes', 'Often'],
    'no_employees': ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000'],
    'benefits': ['No', "Don't know", 'Yes'],
    'care_options': ['No', 'Not sure', 'Yes'],
    'wellness_program': ['No', "Don't know", 'Yes'],
    'seek_help': ['No', "Don't know", 'Yes'],
    'anonymity': ['No', "Don't know", 'Yes'],
    'mental_vs_physical': ['No', "Don't know", 'Yes'],
    'leave': ['Very difficult', 'Somewhat difficult', "Don't know", 'Somewhat easy', 'Very easy'],
    'mental_health_consequence': ['No', 'Maybe', 'Yes'],
    'phys_health_consequence': ['No', 'Maybe', 'Yes'],
    'mental_health_interview': ['No', 'Maybe', 'Yes'],
    'phys_health_interview': ['No', 'Maybe', 'Yes'],
    'coworkers': ['No', 'Some of them', 'Yes'],
    'supervisor': ['No', 'Some of them', 'Yes']
}

binary_columns = [
    "self_employed", "family_history", "remote_work", "tech_company", "obs_consequence"
]

# **Function to collect user input for all features**
def get_user_input():
    user_data = {}

    # Collect input for all features
    for feature in all_features:
        if feature in binary_columns:
            print(f"Choose one for {feature}: Yes/No")  # Show Yes/No choices for binary columns
        elif feature in ordinal_mappings:
            print(f"Choose one for {feature}: {', '.join(ordinal_mappings[feature])}")  # Show ordinal choices

        user_data[feature] = input(f"Enter value for {feature}: ").strip()

    return pd.DataFrame([user_data])

# Get user input
user_df = get_user_input()

# Convert binary categorical columns using the binary mapping
binary_cols = ['self_employed', 'family_history', 'remote_work', 'tech_company', 'obs_consequence']
for col in binary_cols:
    user_df[col] = user_df[col].map(binary_mappings)

# Apply Ordinal Encoding for the ordinal features
user_df[list(ordinal_mappings.keys())] = ordinal_encoder.transform(user_df[list(ordinal_mappings.keys())])

# Ensure Age is numeric
user_df["Age"] = pd.to_numeric(user_df["Age"], errors="coerce")

# Convert to integer type
user_df = user_df.astype(int)

# **Scale all features using the saved scaler**
user_df_scaled = scaler.transform(user_df)

# **Select only the relevant features for prediction**
selected_indices = [user_df.columns.get_loc(col) for col in selected_features]
user_df_scaled_selected = user_df_scaled[:, selected_indices]

# **Make prediction**
prediction = rf_model.predict(user_df_scaled_selected)
prediction_proba = rf_model.predict_proba(user_df_scaled_selected)[:, 1]

def reverse_map_features(input_df):
    reversed_dict = {}

    # Reverse binary encoding (1 -> Yes, 0 -> No)
    for col in binary_columns:
        reversed_dict[col] = input_df[col].map({1: "Yes", 0: "No"})

    # Reverse ordinal encoding
    for col, categories in ordinal_mappings.items():
        reversed_dict[col] = input_df[col].apply(lambda x: categories[int(x)] if pd.notnull(x) else x)

    return pd.DataFrame(reversed_dict)

mapped_input = reverse_map_features(user_df)

# **Filter mapped_input to include only selected features**
mapped_input_selected = mapped_input[selected_features]


# **Generate explanation using Gemini**
prompt = f"""
A person with the following mental health assessment features has been analyzed:

{mapped_input_selected.to_dict(orient='records')[0]}

Based on this assessment, the model predicts that this individual {'will seek treatment' if prediction[0] == 1 else 'will not seek treatment'}.

Provide:
1. A  detailed  and descriptive natural language explanation for the prediction.
2. Suggested coping mechanisms and potential next steps.
"""

response = genai.GenerativeModel('gemini-pro').generate_content(prompt)
llm_explanation = response.text

# Display results
print("\n🔹 **Mental Health Treatment Prediction** 🔹")
print(f"Prediction: {'Will Seek Treatment' if prediction[0] == 1 else 'Will Not Seek Treatment'}")
print(f"Confidence Score: {prediction_proba[0]:.2f}")
print("\n🧠 **Explanation from Gemini:**")
print(llm_explanation)
