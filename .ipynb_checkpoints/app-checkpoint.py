import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="AI Sentiment Analysis System",
    page_icon="🤖",
    layout="centered"
)

# Load saved model pipeline
@st.cache_resource
def load_model():
    return joblib.load("models/sentiment_pipeline.pkl")

model = load_model()

# App Header
st.title("🤖 AI-Powered Sentiment Analysis System")
st.markdown("Enter a customer review below to automatically classify its sentiment (**Positive**, **Negative**, or **Neutral**) along with confidence probabilities.")

# Sidebar info
st.sidebar.header("About the Project")
st.sidebar.info(
    "This system uses a Machine Learning pipeline (TF-IDF Vectorizer + Logistic Regression) "
    "trained on 10,000 customer reviews across multiple product categories and feedback sources."
)
st.sidebar.markdown("---")
st.sidebar.subheader("Sample Reviews to Try:")
st.sidebar.markdown("- *'This product exceeded my expectations in several ways. It has been a good experience so far.'*")
st.sidebar.markdown("- *'The item is uncomfortable to use and poorly designed. I would not buy it again.'*")
st.sidebar.markdown("- *'The product comes with a charging adapter and a short cable.'*")

# Main input form
with st.form("sentiment_form"):
    user_review = st.text_area("Customer Review Text:", placeholder="Type or paste review here...", height=120)
    submitted = st.form_submit_button("Analyze Sentiment")

if submitted:
    if user_review.strip() == "":
        st.warning("Please enter a valid review text before analyzing.")
    else:
        # Prediction
        prediction = model.predict([user_review])[0]
        probabilities = model.predict_proba([user_review])[0]
        classes = model.classes_
        
        # Display results
        st.markdown("---")
        st.subheader("Analysis Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == "Positive":
                st.success(f"### Predicted Sentiment: {prediction} 😊")
            elif prediction == "Negative":
                st.error(f"### Predicted Sentiment: {prediction} 😞")
            else:
                st.warning(f"### Predicted Sentiment: {prediction} 😐")
                
        with col2:
            max_prob = np.max(probabilities)
            st.metric(label="Confidence Score", value=f"{max_prob * 100:.2f}%")
        
        st.markdown("#### Class Probabilities Breakdown:")
        prob_df = pd.DataFrame({
            "Sentiment": classes,
            "Probability": probabilities
        }).sort_values(by="Probability", ascending=False)
        
        for idx, row in prob_df.iterrows():
            st.write(f"**{row['Sentiment']}**: {row['Probability'] * 100:.2f}%")
            st.progress(float(row['Probability']))

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Month 2 NLP & Machine Learning Project | Streamlit Application</div>", unsafe_allow_html=True)