# Mental Health Prediction in Tech Industry:

## Overview
This repository contains the files of a Machine Learning project aimed at predicting mental health treatment-seeking behavior based on various demographic and workplace-related features. It utilizes a **Random Forest model** for prediction and integrates **Google's Gemini LLM** to provide a detailed explanation for the prediction.

## Important Files
- **Mental_Health_Tarun_Behera.ipynb:** This notebook includes the complete analysis of the dataset, covering initial insights, exploratory data analysis (EDA), model testing, LLM Integration, UI development, and model deployment.
- **predict_mental_health.py:** This script handles inference, including model testing and integration with the Large Language Model (LLM).
- **mental_health_ui.py:** This file contains the code for UI development and model deployment.


## Dataset Used
- **Name:** Mental Health in Tech Survey
- **Source:** [Kaggle](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
- **Description:** The dataset contains survey responses related to mental health conditions in the tech industry.

## Initial Insights and Exploratory Data Analysis:

### Initial Analysis
- The dataset has **1259 rows** and **27 columns**.
- **Data types:** All columns are categorical except `Age`, which is numerical.
- **Missing values** found in `state`, `work_interfere`, `comments`, `self_employed`, and `Age`.

### Data Cleaning
- **Invalid Age values** (negative, extremely high/low) replaced with `NaN` and imputed with the median.
- **Dropped irrelevant columns:** `Timestamp`, `Country`, `state`, `comments`, and `Gender`.
- **Missing values handled:**
  - `work_interfere` and `self_employed`: Filled with mode.
  - `Age`: Filled with median.

### Data Exploration & Visualization
- **Univariate analysis** performed on categorical columns using bar plots.
- **Age distribution** analyzed with histogram and boxplot (outliers detected).
- No class imbalance found in the target column `treatment`.
-**Multivariate analysis** performed on both categorical and numerical mainly focused on their relationship with `treatment`.

## Next Steps
- Feature encoding and transformation.
- Model selection and training.
- Inference and UI/CLI development.


## Steps Involved

### 1. Data Preprocessing & Feature Engineering
- **Outlier Handling & Skewness Correction**:
  - Winsorization applied to `Age` column for outliers.
  - Box-Cox transformation used to correct skewness.
- **Encoding Categorical Columns**:
  - Label Encoding for binary categorical variables.
  - Ordinal Encoding for ordered categorical features.
- **Feature Scaling**:
  - StandardScaler applied to  features.

### 2. Feature Selection
- **Random Forest Feature Importance**:
  - Identified the top 13 most important features.
- **Final Selected Features**:
  - `family_history`, `work_interfere`, `no_employees`, `care_options`, `leave`, `benefits`, `coworkers`, `mental_health_consequence`, `phys_health_interview`, `mental_vs_physical`, `supervisor`, `seek_help`, `wellness_program`.

### 3. Model Selection & Training
- **Algorithms Used**:
  - Support Vector Machine (SVM)
  - Random Forest
  - XGBoost
  - Neural Network (Keras)
- **Hyperparameter Tuning**:
  - GridSearchCV applied for Random Forest & XGBoost.
  - Keras Tuner used for Neural Network.

### 4. Model Evaluation
- **Metrics Used**:
  - Accuracy, Precision, Recall, F1-score, ROC-AUC, Cross Validation.
  The following results were obtained:

| Model  | Accuracy | Precision | Recall   | F1 Score | ROC-AUC  |
|-----------------------|----------|-----------|----------|----------|----------|
| Support Vector Machine | 0.706349 | 0.725000  | 0.679688 | 0.701613 | 0.756993 |
| Random Forest         | 0.750000 | 0.773109  | 0.718750 | 0.744939 | 0.808531 |
| XGBoost               | 0.730159 | 0.758621  | 0.687500 | 0.721311 | 0.801222 |
| Neural Network        | 0.734127 | 0.740157  | 0.734375 | 0.737255 | 0.778793 |


| Model                 | F1 Score CV | F1 Score Std |
|-----------------------|-------------|--------------|
| Support Vector Machine | 0.738432    | 0.029814    |
| Random Forest         | 0.762637    | 0.020580     |
| XGBoost               | 0.770760    | 0.023718     |
| Neural Network        | 0.710028    | 0.050284     |


- **Best Model Selection**:
  - The model with the best trade-off between precision and recall is selected- Random Forest
- **Random Forest** has:
  - Highest **Accuracy**: 0.7500
  - High **F1 Score**: 0.7449
  - Best **ROC-AUC**: 0.8085
  - Lowest **Variance in F1 Score** across folds: 0.0206, ensuring stability.
  -  **Conclusion**: Based on the performance metrics, we select the **Random Forest Model** as the final model.


### 5. Deployment
- **Saving Models & Encoders**:
  - The trained model, scaler, and ordinal encoder are saved using `joblib`.
- **Inference Pipeline**:
  - Preprocessing steps implemented to ensure consistency during predictions.

## Model Interpretation

### SHAP (SHapley Additive exPlanations)

To understand the feature contributions in the Random Forest model, SHAP was used:

#### Steps:
- Initialized the SHAP explainer with `TreeExplainer`.
- Computed SHAP values for `X_test_scaled`.
- Handled binary and multi-class cases appropriately.
- Plotted a SHAP summary plot to visualize feature importance.

#### Key Observations:
- **Most Important Features**: `family_history`, `work_interfere`, `care_options`, and `benefits` have the highest SHAP values, meaning they strongly influence the model’s predictions.
- **Less Impactful Features**: `phys_health_interview`, `seek_help`, `mental_vs_physical`, and `wellness_program` have lower SHAP values, indicating minimal impact.
- **Influence of Workplace Factors**: Features like `leave`, `supervisor`, and `coworkers` impact predictions, showing that workplace policies affect mental health outcomes.

### LIME (Local Interpretable Model-Agnostic Explanations)

LIME was used to explain individual predictions.

#### Steps:
- Initialized `LimeTabularExplainer` with scaled training data.
- Selected a test instance and generated explanations using LIME.
- Displayed the explanation in a notebook.

#### Key Observations:
- **Prediction Breakdown**: Example instance predicted `No Treatment` with **57% confidence** and `Treatment` with **43% confidence**.
- **Feature Influence**:
  - `leave` increased the likelihood of **Treatment**.
  - `family_history` increased the likelihood of **No Treatment**.
- **Visual Interpretation**: Blue bars push predictions toward `No Treatment`, while orange bars push toward `Treatment`.


## Mental Health Prediction with LLM Explanation

- The code is can be found in the Inference Script: **predict_mental_health.py**.

## Overview
 It utilizes a **Random Forest model** for prediction and integrates **Google's Gemini LLM** to provide a detailed explanation for the prediction.

## Features
- **Pre-trained Random Forest Model**: Used for binary classification (whether an individual will seek treatment or not).
- **Feature Processing**:
  - Binary categorical variables are mapped to 0/1.
  - Ordinal categorical variables are encoded using a pre-trained ordinal encoder.
  - Numerical features are standardized using a pre-trained scaler.
-**Model Prediction:**
  - The Random Forest ML model provides predictions based on the user's input.
- **LLM Explanation**:
  - Uses **Google Gemini (gemini-1.5-pro)** to generate natural language explanations for predictions.
  - Provides insights into the model's decision-making process.
  - Suggests coping mechanisms and next steps for the individual.

## Dependencies
- `numpy`
- `pandas`
- `joblib`
- `scikit-learn`
- `google.generativeai`
- `warnings`

## Files
- **`predict_mental_health.py`**: Main inference script.
- **`random_forest_model.pkl`**: Trained Random Forest model.
- **`scaler.pkl`**: Pre-trained standard scaler for feature scaling.
- **`ordinal_encoder.pkl`**: Pre-trained ordinal encoder for categorical features.

## How to Use
1. **Run the script ( in the colab notebook) as follows:**:
   ```bash
   !python predict_mental_health.py

🔹 Mental Health Treatment Prediction 🔹

Prediction: Will Seek Treatment
Confidence Score: 0.85

🧠 Explanation from Gemini:

Based on the input features, the model predicts that the individual is likely to seek treatment due to factors such as a history of mental health issues, workplace benefits, and accessibility to care options. Suggested next steps include...

## Mental Health Prediction UI

The file: **mental_health_ui.py** contains a **Streamlit-based UI** for predicting whether an individual is likely to seek **mental health treatment** based on responses to various questions. The UI utilizes a **Random Forest model** for predictions and **Google Gemini AI** for providing an explanation of the prediction.

## Features

- **User-Friendly UI:** Built with **Streamlit** for easy interaction.
- **Machine Learning Model:** Utilizes a trained **Random Forest** model for predictions.
- **Feature Selection & Encoding:** Preprocesses user input using **encoding techniques** and **standard scaling** followed by feature selection.
- **AI Explanation:** Uses **Google Gemini AI** to provide insights on the prediction.
- **Local Deployment:** Can be run locally using **Streamlit** and **LocalTunnel**.

## How to use:

1. Install required dependencies:
   ```bash
   !pip install -q streamlit pandas numpy scikit-learn joblib google-generativeai

2. Install LocalTunnel ( in colab):
   ```bash
   !npm local tunnel

## Run the Streamlit Application
1. To run the application ( first obtain the password for the Tunnel):
	```bash
		 !wget -q -O - ipv4.icanhazip.com
2.  To expose the app over the internet using LocalTunnel:
	```bash
		!streamlit run mental_health_ui.py & npx localtunnel --port 8501

The above will provide a url to view our Streamlit app in the browser. First, you will be required to input the password for the Tunnel. Use the password obtained in Step 1. Then the Streamlit application can be used.


# How It Works

## User Input
Users provide answers to a set of mental health-related questions.

## Preprocessing
- **Binary features** are converted to numerical values.
- **Ordinal features** are mapped using predefined categories.
- **Age** is converted to numeric format.
- Data is **scaled** using a pre-trained scaler.
- Feature selection is performed and the selected features are given to the model for prediction.

## Prediction
The **Random Forest** model predicts whether the user is likely to seek mental health treatment.

## AI Explanation
The **Google Gemini AI** generates a natural language explanation of the prediction along with suggested coping mechanisms.

# Dependencies
- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Google Generative AI
- LocalTunnel










