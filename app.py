# app.py
import streamlit as st
import pandas as pd
import pickle

# Load the trained model and label encoders
with open('student_grade_predictor.pkl', 'rb') as f:
    model, label_encoders = pickle.load(f)

# Streamlit setup
st.set_page_config(page_title="Student Grade Predictor", page_icon="🎓", layout="centered")
st.title("🎓 Student Grade Predictor")
st.markdown("Predict a student's final grade (G3) based on their profile.")

# Sidebar input
st.sidebar.header("📋 Enter Student Information")
user_input = {}

# Collect categorical input
for column in label_encoders.keys():
    options = list(label_encoders[column].classes_)
    user_input[column] = st.sidebar.selectbox(f"{column.replace('_', ' ').title()}:", options)

# Collect numerical input
numerical_cols = [col for col in model.feature_names_in_ if col not in label_encoders]
for col in numerical_cols:
    user_input[col] = st.sidebar.number_input(f"{col.replace('_', ' ').title()}:", step=1.0)

# When Predict button is clicked
if st.sidebar.button("🔮 Predict Grade"):
    input_df = pd.DataFrame([user_input])

    # Encode categorical features
    for col, le in label_encoders.items():
        input_df[col] = le.transform([input_df[col][0]])

    # Ensure correct column order
    input_df = input_df[model.feature_names_in_]

    # Prediction
    prediction = model.predict(input_df)
    st.success(f"✅ Predicted Final Grade: **{prediction[0]}** 🎯")

# Footer
st.markdown("---")
st.markdown("<center>Made with ❤️ using Streamlit & RandomForest</center>", unsafe_allow_html=True)
