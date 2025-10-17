# app.py

import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# ✅ Must be the first Streamlit command
# -------------------------------
st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")

# -------------------------------
# Load the saved model
# -------------------------------
@st.cache_resource
def load_model():
    with open('naive_bayes_sms_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# -------------------------------
# Streamlit Interface
# -------------------------------
st.markdown("""
    <style>
        body {
            background-color: #f9fafc;
        }
        .main-title {
            text-align: center;
            color: #2b5876;
            font-size: 2.2em;
            font-weight: bold;
        }
        .prediction-box {
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            font-size: 1.3em;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<p class='main-title'>📩 SMS Spam Detection App</p>", unsafe_allow_html=True)
st.markdown("""
Enter the SMS details in the sidebar and the model will predict whether it is **Spam** or **Not Spam**.
""")

# Sidebar inputs
st.sidebar.header("📨 Enter SMS Features")
word_free = st.sidebar.number_input("Count of word 'free'", min_value=0, value=1, step=1)
word_win = st.sidebar.number_input("Count of word 'win'", min_value=0, value=0, step=1)
word_offer = st.sidebar.number_input("Count of word 'offer'", min_value=0, value=0, step=1)
sms_len = st.sidebar.number_input("SMS Length (characters)", min_value=1, value=20, step=1)

st.markdown("---")

# Button for prediction
if st.button("🔍 Predict Spam"):
    new_sms = pd.DataFrame({
        'word_freq_free': [word_free],
        'word_freq_win': [word_win],
        'word_freq_offer': [word_offer],
        'sms_length': [sms_len]
    })
    
    prediction = model.predict(new_sms)
    
    if prediction[0] == 1:
        st.markdown("<div class='prediction-box' style='background-color:#ffe6e6; color:#b30000;'>🚨 This SMS is Spam</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='prediction-box' style='background-color:#e6ffe6; color:#007a00;'>✅ This SMS is Not Spam</div>", unsafe_allow_html=True)

st.markdown("---")
st.info("💡 Tip: Spam messages often contain words like 'free', 'win', or 'offer' several times, and are usually longer than normal texts.")
