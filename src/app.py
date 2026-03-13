import streamlit as st
import pickle
import numpy as np
import requests

# -----------------------------
# Load model and encoders
# -----------------------------
@st.cache_resource
def load_resources():

    with open("src/trained_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("src/drug_encoder.pkl", "rb") as f:
        drug_encoder = pickle.load(f)

    with open("src/side_effect_encoder.pkl", "rb") as f:
        side_effect_encoder = pickle.load(f)

    return model, drug_encoder, side_effect_encoder


model, drug_encoder, side_effect_encoder = load_resources()

drug_options = list(drug_encoder.classes_)

# -----------------------------
# AI explanation function
# -----------------------------
def get_explanation(side_effect, language):

    try:

        api_key = st.secrets["GROQ_API_KEY"]

        url = "https://api.groq.com/openai/v1/chat/completions"

        prompt = f"Explain the medical side effect '{side_effect}' in one simple sentence in {language}."

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 80
        }

        response = requests.post(url, headers=headers, json=payload)

        data = response.json()

        # show API response for debugging
        if "choices" in data:
            return data["choices"][0]["message"]["content"]

        return f"API Response Error: {data}"

    except Exception as e:

        return f"Error: {str(e)}"


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="ADR Predictor",
    page_icon="💊",
    layout="wide"
)

st.title("💊 Pharmacogenomics ADR Predictor")

st.markdown(
    "Predict potential side effects based on drug and genetic score with AI-powered explanations."
)

# Sidebar
with st.sidebar:

    st.header("Settings")

    language = st.selectbox(
        "Explanation Language",
        ["English", "Hindi", "Marathi"]
    )

    if "GROQ_API_KEY" in st.secrets:
        st.success("Groq API key detected")
    else:
        st.error("Groq API key not found")


# Layout
col1, col2 = st.columns([1, 1])


# Input
with col1:

    st.subheader("Input Parameters")

    drug_name = st.selectbox("Select Drug", drug_options)

    genetic_score = st.number_input(
        "Genetic Score",
        min_value=0,
        max_value=1000,
        value=100,
        step=1
    )


# Prediction
with col2:

    st.subheader("Prediction Results")

    if st.button("🔍 Predict Side Effects", type="primary"):

        drug_encoded = drug_encoder.transform([drug_name])[0]

        input_data = np.array([[drug_encoded, genetic_score]])

        proba = model.predict_proba(input_data)[0]

        top_indices = np.argsort(proba)[-5:][::-1]

        top_side_effects = side_effect_encoder.inverse_transform(top_indices)

        st.success(
            f"Top 5 predicted side effects for {drug_name} with genetic score {genetic_score}:"
        )

        for i, effect in enumerate(top_side_effects, 1):

            explanation = get_explanation(effect, language)

            st.markdown(f"**{i}. {effect}**  \n{explanation}")

            st.divider()