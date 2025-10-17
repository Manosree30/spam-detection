# spam-detection
# SMS Spam Detection Using Naive Bayes
## Project Overview

This project demonstrates how to build a SMS spam detection system using a Naive Bayes classifier. The system classifies SMS messages as spam or not spam based on the frequency of certain keywords and message length.


## Features Used

The following features are extracted from each SMS message:

word_freq_free: Number of times the word "free" appears in the message

word_freq_win: Number of times the word "win" appears

word_freq_offer: Number of times the word "offer" appears

sms_length: Total number of characters in the message

### Target variable:

is_spam → 1 for spam, 0 for not spam

## Steps Performed

Data Loading & Preprocessing

Read the CSV dataset

Map labels (ham → 0, spam → 1)

Extract word frequencies and message length

## Exploratory Data Analysis (EDA)

Checked class distribution (spam vs ham)

Analyzed message length

Examined correlations between features

## Model Training

Split dataset into training and testing sets

Trained a Gaussian Naive Bayes model using the extracted features

## Evaluation

Evaluated the model using accuracy and classification report

Achieved high accuracy in detecting spam messages

## Prediction

Tested the model with a sample SMS

Predicts whether a message is spam or not

## Saving the Model

Saved the trained model using pickle (naive_bayes_sms_model.pkl)

Model can be loaded later without retraining

## Deployment 

A Streamlit app can be used for interactive SMS spam detection

## Requirements

Python 3.x

pandas

scikit-learn

matplotlib (for EDA)

seaborn (for EDA)

streamlit (for deployment)

pickle (for saving/loading the model)

## Install packages using:

pip install pandas scikit-learn matplotlib seaborn streamlit

## Usage
1. Running the Jupyter Notebook

Open SMS_Spam_Detection.ipynb

Run all cells to train and evaluate the model

Test predictions with custom SMS messages

2. Using the Pickle Model
import pickle
import pandas as pd

# Load the saved model
with open('naive_bayes_sms_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Example prediction
new_sms = pd.DataFrame({'word_freq_free':[2],'word_freq_win':[1],'word_freq_offer':[3],'sms_length':[70]})
prediction = model.predict(new_sms)
print("Spam" if prediction[0]==1 else "Not Spam")

3. Running Streamlit App (Optional)
streamlit run app.py


Enter the word counts and SMS length to predict spam messages interactively.

## Conclusion

Naive Bayes works well for text classification tasks like spam detection

The model is fast, requires less training data, and handles categorical/numeric features effectively

## Limitations:

Assumes independence between features

Probability scores may not be perfectly calibrated

