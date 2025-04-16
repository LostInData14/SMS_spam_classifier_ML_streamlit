import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained pipeline (tuned Random Forest model with TfidfVectorizer)
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
model = pickle.load(open('best_model.pkl','rb'))

# App title
st.title( "SMS Spam Classifier")
st.subheader("Enter a message to know whether it classified as Spam or Not Spam")

# Input box for user input
user_input = st.text_area("Enter SMS text:", "")

# Classify when the user clicks the button
if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        # Predict using the model
        prediction = model.predict([user_input])[0]
        prediction_proba = model.predict_proba([user_input])[0]

        prediction_label = "Spam" if prediction == 1 else "Not Spam"
        confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]

        st.success(f"Prediction: **{prediction_label}**")
        st.write(f"Confidence: {confidence * 100:.2f}%")

        # Show a chart for prediction confidence
        proba_df = pd.DataFrame({
            "Category": ["Not Spam", "Spam"],
            "Confidence": [prediction_proba[0], prediction_proba[1]]
        })
        st.bar_chart(proba_df.set_index('Category'))
