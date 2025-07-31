import streamlit as st
from model import ObesityModel

st.set_page_config(page_title="Obesity Prediction App", page_icon="🍔", layout="centered")

# Load model
try:
    obesity_model = ObesityModel()
except Exception as e:
    st.error(f"Failed to load or process dataset: {e}")
    st.stop()

# Custom CSS and Title
st.markdown("""
    <style>
        .title { text-align: center; font-size: 36px; font-weight: bold; color: #e74c3c; }
        .stButton>button { background-color: #e74c3c; color: white; font-size: 20px; border-radius: 8px; padding: 10px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<p class='title'>Obesity Prediction App 🍔</p>", unsafe_allow_html=True)
st.write("Fill in the details below to predict obesity level.")

# Show model metrics
metrics = obesity_model.get_metrics()
st.subheader("Model Evaluation Metrics")
st.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
st.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
st.metric("Recall", f"{metrics['recall'] * 100:.2f}%")

# Input form
st.subheader("Enter Patient Details")
input_data = {}
col1, col2 = st.columns(2)

label_encoders = obesity_model.get_label_encoders()
feature_names = obesity_model.get_feature_names()

for i, col in enumerate(feature_names):
    with col1 if i % 2 == 0 else col2:
        if col in label_encoders:
            input_data[col] = st.selectbox(col, label_encoders[col].classes_)
        else:
            input_data[col] = st.text_input(col)

# Prediction logic
if st.button("🔍 Predict"):
    try:
        input_list = []
        for col in feature_names:
            val = input_data[col]
            if val.strip() == "":
                st.error(f"Please enter a value for '{col}'.")
                st.stop()

            if col in label_encoders:
                val = label_encoders[col].transform([val])[0]
            else:
                try:
                    val = float(val)
                except ValueError:
                    st.error(f"Invalid input for '{col}'. Please enter a numeric value.")
                    st.stop()

            input_list.append(val)

        prediction = obesity_model.predict(input_list)
        result = obesity_model.get_target_encoder().inverse_transform([prediction])[0]
        st.success(f"### Predicted Obesity Level: {result}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
