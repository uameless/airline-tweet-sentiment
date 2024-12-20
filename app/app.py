import time
import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification
import joblib
import os

model_path_1 = "./models/bert_saved"
model_path_2 = "./models/distilbert_saved"
model_path_3 = "./models/svm"
model_path_4 = "./models/logistic_reg"
vec_folder = "./models/tf_idf"

model_1 = BertForSequenceClassification.from_pretrained(model_path_1)
tokenizer_1 = BertTokenizer.from_pretrained(model_path_1)

model_2 = DistilBertForSequenceClassification.from_pretrained(model_path_2)
tokenizer_2 = DistilBertTokenizer.from_pretrained(model_path_2)

vectorizer = joblib.load(os.path.join(vec_folder, 'vectorizer.pkl'))
svm_model = joblib.load(os.path.join(model_path_3, 'svm_model.pkl'))
lr_model = joblib.load(os.path.join(model_path_4, 'lr_model.pkl'))

def predict_sentiment(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    predicted_class = torch.argmax(logits, dim=1).item()
    
    if predicted_class == 0:
        return "Negative"
    elif predicted_class == 1:
        return "Neutral"
    else:
        return "Positive"
    
def predict_sentiment_ml(model, vectorizer, text):
    vectorized_text = vectorizer.transform([text])

    predicted_class = model.predict(vectorized_text)

    if predicted_class == -1:
        return "Negative"
    elif predicted_class == 0:
        return "Neutral"
    else:
        return "Positive"



st.markdown("<h1 style='color: white; text-align: center;'>Sentiment Analysis App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; font-size: 18px;'>Enter a tweet to analyze sentiment using BERT, DistilBERT, SVM, and Logistic Regression models.</p>", unsafe_allow_html=True)

input_text = st.text_area("Enter a tweet to analyze sentiment:")

if st.button("Predict Sentiment"):
    if input_text:
        with st.spinner("Analyzing sentiment..."):
            time.sleep(2)  
            sentiment_1 = predict_sentiment(model_1, tokenizer_1, input_text)
            sentiment_2 = predict_sentiment(model_2, tokenizer_2, input_text)
            sentiment_3 = predict_sentiment_ml(svm_model, vectorizer, input_text)
            sentiment_4 = predict_sentiment_ml(lr_model, vectorizer, input_text)
        
        def style_sentiment(model_name, sentiment):
            if sentiment == "Positive":
                sentiment_colored = f"<span style='color: green; font-weight: bold;'>{sentiment}</span>"
                icon = "🟢"
            elif sentiment == "Neutral":
                sentiment_colored = f"<span style='color: orange; font-weight: bold;'>{sentiment}</span>"
                icon = "🟠"
            else:
                sentiment_colored = f"<span style='color: red; font-weight: bold;'>{sentiment}</span>"
                icon = "🔴"
            
            return f"""
            <div style='padding: 10px; margin-bottom: 10px; border: 2px solid #ddd; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);'>
                <p style='font-size: 20px; font-weight: normal; display: flex; align-items: center;'>
                    <span style='margin-right: 10px; font-size: 24px;'>{icon}</span>{model_name}: {sentiment_colored}
                </p>
            </div>
            """
        
        st.markdown("<h2>Sentiment Analysis Results</h2>", unsafe_allow_html=True)
        st.markdown(style_sentiment("BERT Model", sentiment_1), unsafe_allow_html=True)
        st.markdown(style_sentiment("DistilBERT Model", sentiment_2), unsafe_allow_html=True)
        st.markdown(style_sentiment("SVM Model", sentiment_3), unsafe_allow_html=True)
        st.markdown(style_sentiment("Logistic Regression Model", sentiment_4), unsafe_allow_html=True)
    else:
        st.write("Please enter a tweet.")
