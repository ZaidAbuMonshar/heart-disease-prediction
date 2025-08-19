# Heart Disease Prediction Project

## Overview
This project aims to develop a machine learning model to predict the presence of heart disease in patients based on various health-related features. The dataset includes attributes such as age, gender, chest pain type, resting blood pressure, cholesterol levels, and more. The model is trained using logistic regression and evaluated using metrics like accuracy, precision, recall, and F1-score.

## Dataset
The dataset contains the following features:
- `age`: Patient's age.
- `sex`: Gender (1 = Male, 0 = Female).
- `cp`: Chest pain type (1-4).
- `trestbps`: Resting blood pressure (mm Hg).
- `chol`: Serum cholesterol level (mg/dl).
- `fbs`: Fasting blood sugar > 120 mg/dl (1 = True, 0 = False).
- `restecg`: Resting electrocardiographic results (0-2).
- `thalach`: Maximum heart rate achieved.
- `exang`: Exercise-induced angina (1 = Yes, 0 = No).
- `oldpeak`: ST depression induced by exercise relative to rest.
- `slope`: Slope of the peak exercise ST segment.
- `ca`: Number of major vessels colored by fluoroscopy.
- `thal`: Thalassemia type (1-3).
- `target`: Presence of heart disease (1 = Disease, 0 = No Disease).

## Data Preprocessing
1. **Handling Missing Values**: The dataset was checked for null values, and none were found.
2. **Removing Duplicates**: A duplicate record was identified and removed to prevent overfitting.
3. **Feature Scaling**: Features were standardized using `StandardScaler` to ensure uniformity.

## Model Training
- **Algorithm Used**: Logistic Regression.
- **Data Split**: The dataset was split into training (80%) and testing (20%) sets.
- **Training**: The model was trained on the scaled training data.

## Evaluation Metrics
The model's performance was evaluated using:
- **Accuracy**: 82%.
- **Confusion Matrix**:
  - True Positives (TP): 24
  - False Positives (FP): 5
  - False Negatives (FN): 6
  - True Negatives (TN): 28
- **Precision-Recall Curve**: Average Precision (AP) = 0.94.
- **ROC Curve**: AUC = 0.92 (excellent performance).

## Usage
To predict heart disease for new patients:
1. Prepare the patient data as a DataFrame with the same features as the training data.
2. Scale the data using the same `StandardScaler` instance.
3. Use the trained model to predict the presence of heart disease and the probability.

Example:
```python
new_patient = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]
new_patient_scaled = scaler.transform(new_patient)
prediction = model.predict(new_patient_scaled)
probability = model.predict_proba(new_patient_scaled)
print(f"Prediction: {prediction[0]}, Probability: {probability[0][1]:.4f}")
