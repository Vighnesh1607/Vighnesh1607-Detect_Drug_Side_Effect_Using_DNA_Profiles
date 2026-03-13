import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests
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
# Basic fallback explanations for common side effects (used if no API key).
EXPLANATION_FALLBACK = {
    "Alopecia": {
        "English": "Alopecia is hair loss that can occur as a side effect of some medications.",
        "Hindi": "एलोपीशिया दवाओं के कुछ दुष्प्रभावों के रूप में बालों का झड़ना है।",
        "Marathi": "एलोपीशिया ही काही औषधांच्या दुष्परिणामांमुळे होणारी केस गळती आहे."
    },
    "Nausea": {
        "English": "Nausea is a feeling of sickness in the stomach that often comes before vomiting.",
        "Hindi": "मतली पेट में बेचैनी की भावना है जो अक्सर उल्टी से पहले होती है।",
        "Marathi": "उलट्या येण्यापूर्वी उदरात अस्वस्थता ही मळमळ म्हणून ओळखली जाते."
    },
    "Headache": {
        "English": "Headache is pain in the head or neck area and can be triggered by many drugs.",
        "Hindi": "सिर दर्द सिर या गर्दन के क्षेत्र में दर्द है और कई दवाओं से हो सकता है।",
        "Marathi": "डोकेदुखी म्हणजे डोक्‍यात किंवा मानेमध्ये वेदना आणि अनेक औषधांमुळे होऊ शकते."
    },
    "Fatigue": {
        "English": "Fatigue is a feeling of extreme tiredness and lack of energy.",
        "Hindi": "थकान अत्यधिक थकावट और ऊर्जा की कमी की भावना है।",
        "Marathi": "थकवा ही तीव्र थकवाट आणि उर्जेची कमतरता आहे."
    },
    "Dizziness": {
        "English": "Dizziness is a sensation of being unbalanced or lightheaded.",
        "Hindi": "चक्कर आना एक असंतुलित या हल्का महसूस करने की भावना है।",
        "Marathi": "चक्कर येणे म्हणजे असमाधानकारक किंवा सौम्य वाटण्याची भावना आहे."
    },
    "Hyperhidrosis": {
        "English": "Hyperhidrosis is excessive sweating which can be triggered by certain medications.",
        "Hindi": "हाइपरहाइड्रोसिस अत्यधिक पसीना है जो कुछ दवाओं के कारण हो सकता है।",
        "Marathi": "हायपरहायड्रोसिस म्हणजे जास्त प्रमाणात घाम येणे, जे काही औषधांमुळे होऊ शकते."
    },
    "Quadriparesis": {
        "English": "Quadriparesis is weakness in all four limbs, which can sometimes occur with certain drugs.",
        "Hindi": "क्वाड्रिपारेसिस सभी चार अंगों में कमजोरी है, जो कभी-कभी कुछ दवाओं के साथ हो सकती है।",
        "Marathi": "क्वाड्रिपॅरेसिस हा सर्व चार अंगात कमजोरपणा आहे, जो काही औषधांमुळे होऊ शकतो."
    },
    "Asterixis": {
        "English": "Asterixis is a flapping tremor of the hands that can appear with certain neurological side effects.",
        "Hindi": "एस्टेरिक्सिस हाथों में एक फ्लैपिंग कंपन है जो कुछ न्यूरोलॉजिकल दुष्प्रभावों के साथ दिखाई दे सकता है।",
        "Marathi": "एस्टेरिक्सिस म्हणजे हातातील फडफडणारा थरथरणा आहे जो काही न्यूरोलॉजिकल दुष्परिणामांसह दिसू शकतो."
    },
    "Dermatitis atopic": {
        "English": "Atopic dermatitis is a chronic skin condition that can be worsened by some medications.",
        "Hindi": "एटोपिक डर्मेटाइटिस एक पुरानी त्वचा की स्थिति है जिसे कुछ दवाएं बिगाड़ सकती हैं।",
        "Marathi": "अॅटोपिक डर्माटायटिस ही एक दीर्घकालीन त्वचेची स्थिती आहे जी काही औषधांनी बिघडू शकते."
    },
    "Drug screen positive": {
        "English": "A positive drug screen means a test may show the presence of certain substances, which can sometimes happen due to medications.",
        "Hindi": "एक सकारात्मक ड्रग स्क्रीन का मतलब है कि परीक्षण में कुछ पदार्थ मौजूद हो सकते हैं, जो कभी-कभी दवाओं के कारण होता है।",
        "Marathi": "एक सकारात्मक औषध तपासणी म्हणजे काही पदार्थांचा शोध लागू शकतो, जे कधीकधी औषधांमुळे होते."
    },
}


def get_api_key():
    # Streamlit secrets are stored in st.secrets, not env vars, so check both.
    return os.getenv('GROQ_API_KEY') or st.secrets.get('GROQ_API_KEY')


def _fallback_explanation(side_effect, language):
    side_fallback = EXPLANATION_FALLBACK.get(side_effect, {})
    if side_fallback:
        return side_fallback.get(language, side_fallback.get('English'))

    # Generic fallback language template
    generic_templates = {
        "English": f"{side_effect} is a reported side effect for some medications.",
        "Hindi": f"{side_effect} कुछ दवाओं के लिए एक रिपोर्ट किया गया दुष्प्रभाव है।",
        "Marathi": f"{side_effect} काही औषधांशी संबंधित अहवालित दुष्परिणाम आहे."
    }
    return generic_templates.get(language, generic_templates['English'])


def get_explanation(side_effect, language):
    api_key = get_api_key()
    if not api_key:
        return _fallback_explanation(side_effect, language)

    prompt = f"Explain the medical side effect '{side_effect}' in one short sentence in {language}."
    url = "https://api.groq.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()

        # Ensure the response contains the expected structure
        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        if content and isinstance(content, str):
            return content.strip()

        # Unexpected structure: fall back
        return _fallback_explanation(side_effect, language)
    except Exception as e:
        # If API fails, show a brief error footnote and fall back
        return f"(AI explanation unavailable: {str(e)})\n" + _fallback_explanation(side_effect, language)

# UI
st.set_page_config(page_title="ADR Predictor", page_icon="💊", layout="wide")
st.title("💊 Pharmacogenomics ADR Predictor")
st.markdown("Predict potential side effects based on drug and genetic score with AI-powered explanations.")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    language = st.selectbox("Explanation Language", ["English", "Hindi", "Marathi"])
    enable_explanations = st.checkbox("Enable AI Explanations", value=True)

    if get_api_key():
        st.success("Groq API key detected — AI explanations are enabled.")
    else:
        st.warning(
            "No Groq API key found — using built-in fallback explanations. "
            "To enable AI explanations, add a Groq key in Streamlit Secrets (Settings → Secrets) using key `GROQ_API_KEY`."
        )

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Parameters")
    drug_name = st.selectbox("Select Drug", drug_options)
    genetic_score = st.number_input("Genetic Score", min_value=0, max_value=1000, value=100, step=1)

with col2:
    st.subheader("Prediction Results")
    if st.button("🔍 Predict Side Effects", type="primary"):
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