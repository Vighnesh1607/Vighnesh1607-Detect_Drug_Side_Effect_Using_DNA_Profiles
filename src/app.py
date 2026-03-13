import streamlit as st
import pandas as pd
import pickle
import numpy as np
from groq import Groq
import os

# Load model and encoders
@st.cache_resource
def load_resources():
    with open('src/trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('src/drug_encoder.pkl', 'rb') as f:
        drug_encoder = pickle.load(f)
    with open('src/side_effect_encoder.pkl', 'rb') as f:
        side_effect_encoder = pickle.load(f)
    return model, drug_encoder, side_effect_encoder

model, drug_encoder, side_effect_encoder = load_resources()

# Get unique drugs for dropdown
drug_options = list(drug_encoder.classes_)

# Function to get explanation from Groq
def get_explanation(side_effect, language):
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        return "API key not set. Please set GROQ_API_KEY environment variable."
    
    client = Groq(api_key=api_key)
    prompt = f"Explain the medical side effect '{side_effect}' in one short sentence in {language}."
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",  # Best model for accuracy
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Explanation not available: {str(e)}"

# UI
st.set_page_config(page_title="ADR Predictor", page_icon="💊", layout="wide")
st.title("💊 Pharmacogenomics ADR Predictor")
st.markdown("Predict potential side effects based on drug and genetic score with AI-powered explanations.")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    language = st.selectbox("Explanation Language", ["English", "Hindi", "Marathi"])
    enable_explanations = st.checkbox("Enable AI Explanations", value=True)
    if not os.getenv('GROQ_API_KEY'):
        st.warning("GROQ_API_KEY environment variable not set. Please set it before running.")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Parameters")
    drug_name = st.selectbox("Select Drug", drug_options)
    genetic_score = st.number_input("Genetic Score", min_value=0, max_value=1000, value=100, step=1)

with col2:
    st.subheader("Prediction Results")
    if st.button("🔍 Predict Side Effects", type="primary"):
        if enable_explanations and not os.getenv('GROQ_API_KEY'):
            st.error("GROQ_API_KEY environment variable not set. Please set it and restart the app.")
        else:
            # Encode input
            drug_encoded = drug_encoder.transform([drug_name])[0]

            # Prepare input
            input_data = np.array([[drug_encoded, genetic_score]])

            # Get probabilities
            proba = model.predict_proba(input_data)[0]

            # Get top 5 indices
            top_indices = np.argsort(proba)[-5:][::-1]

            # Decode to names
            top_side_effects = side_effect_encoder.inverse_transform(top_indices)

            # Display results
            st.success(f"Top 5 predicted side effects for {drug_name} with genetic score {genetic_score}:")
            
            for i, effect in enumerate(top_side_effects, 1):
                if enable_explanations:
                    explanation = get_explanation(effect, language)
                    st.markdown(f"**{i}. {effect}**  \n{explanation}")
                else:
                    st.markdown(f"**{i}. {effect}**")
                st.divider()