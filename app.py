import streamlit as st
import joblib
import os
import pandas as pd

# Load semua model
model_paths = {
    'Logistic Regression (Optimized)': 'model/logistic_regression_(optimized).pkl',
    'Random Forest (Optimized)': 'model/random_forest_(optimized).pkl',
    'SVM (Calibrated)': 'model/svm_(calibrated).pkl',
    'Neural Network': 'model/neural_network.pkl',
    'Voting Ensemble': 'model/voting_ensemble.pkl',
    'Stacking Ensemble (Advanced)': 'model/stacking_ensemble_(advanced).pkl',
    'Stacking Model': 'model/stacking_model.pkl'
}
models = {name: joblib.load(path) for name, path in model_paths.items()}

# Load semua encoder
encoder_dir = 'encoders'
encoders = {}
for filename in os.listdir(encoder_dir):
    if filename.endswith('_encoder.pkl'):
        feature_name = filename.replace('_encoder.pkl', '')
        encoders[feature_name] = joblib.load(os.path.join(encoder_dir, filename))

# UI Streamlit
st.title("📘 Student Performance Prediction")

with st.sidebar:
    st.header("Pilih Model Machine Learning")
    selected_model_name = st.selectbox("Model", list(models.keys()))
    selected_model = models[selected_model_name]
    st.success(f'Model: **{selected_model_name}**')

st.subheader("📥 Input Data Students")

# Fungsi untuk mengambil kelas dari encoder
def get_valid_choices(feature_name):
    encoder = encoders.get(feature_name)
    return encoder.classes_.tolist() if encoder else []

# Input pengguna
tuition = st.selectbox("Tuition Fees Up To Date", get_valid_choices('Tuition_fees_up_to_date'))
scholar = st.selectbox("Scholarship Holder", get_valid_choices('Scholarship_holder'))
course = st.selectbox("Course", get_valid_choices('Course'))
debtor = st.selectbox("Debtor", get_valid_choices('Debtor'))
gender = st.selectbox("Gender", get_valid_choices('Gender'))
app_mode = st.selectbox("Application Mode", get_valid_choices('Application_mode'))
sem_approved = st.selectbox("Curricular Units 1st Sem Approved", get_valid_choices('Curricular_units_1st_sem_approved'))
age_enroll = st.selectbox("Age at Enrollment", get_valid_choices('Age_at_enrollment'))
sem_enrolled = st.selectbox("Curricular Units 1st Sem Enrolled", get_valid_choices('Curricular_units_1st_sem_enrolled'))
admission_grade = st.selectbox("Admission Grade", get_valid_choices('Admission_grade'))
previous_grade = st.selectbox("Previous Qualification Grade", get_valid_choices('Previous_qualification_grade'))

# Tombol Prediksi
if st.button("🎯 Prediksi"):
    input_data = pd.DataFrame([{
        'Tuition_fees_up_to_date': tuition,
        'Scholarship_holder': scholar,
        'Course': course,
        'Debtor': debtor,
        'Gender': gender,
        'Application_mode': app_mode,
        'Curricular_units_1st_sem_approved': sem_approved,
        'Age_at_enrollment': age_enroll,
        'Curricular_units_1st_sem_enrolled': sem_enrolled,
        'Admission_grade': admission_grade,
        'Previous_qualification_grade': previous_grade
    }])

    # Urutkan kolom sesuai yang dibutuhkan model
    try:
        feature_order = selected_model.feature_names_in_
        input_data = input_data[feature_order]
    except AttributeError:
        pass  # Model tidak menyimpan feature_names_in_ (misalnya neural network)

    # Encode fitur kategorikal
    for col in input_data.columns:
        if col in encoders:
            try:
                input_data[col] = encoders[col].transform(input_data[col])
            except Exception as e:
                st.error(f"Encoding error pada fitur '{col}': {e}")
                st.stop()

    # Prediksi
    status_mapping = {
    0: 'Dropout',
    1: 'Enrolled',
    2: 'Graduate'
    }
    prediction = selected_model.predict(input_data)
    label = status_mapping.get(prediction[0], 'Unknown')
    st.success(f"Hasil Prediksi: **{label}**")
