# app.py

import streamlit as st
import pandas as pd
import pickle

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
st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")

st.title("📩 SMS Spam Detection App")
st.markdown("""
Enter the SMS details in the sidebar and the model will predict if it is **Spam** or **Not Spam**.
""")

# Sidebar inputs
st.sidebar.header("📨 Enter SMS Features")
word_free = st.sidebar.number_input("Count of word 'free'", min_value=0, value=1, step=1)
word_win = st.sidebar.number_input("Count of word 'win'", min_value=0, value=0, step=1)
word_offer = st.sidebar.number_input("Count of word 'offer'", min_value=0, value=0, step=1)
sms_len = st.sidebar.number_input("SMS Length (characters)", min_value=1, value=20, step=1)

st.markdown("---")

# Button for prediction
if st.button("Predict Spam"):
    new_sms = pd.DataFrame({
        'word_freq_free': [word_free],
        'word_freq_win': [word_win],
        'word_freq_offer': [word_offer],
        'sms_length': [sms_len]
    })
    
    prediction = model.predict(new_sms)
    
    if prediction[0] == 1:
        st.markdown("<h2 style='color:red;'>🚨 This SMS is Spam</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color:green;'>✅ This SMS is Not Spam</h2>", unsafe_allow_html=True)

st.markdown("---")
st.info("💡 Tip: Spam messages often contain words like 'free', 'win', or 'offer' multiple times, and may be longer than normal messages.")
