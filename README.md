# AI-Powered Sentiment Analysis System (Month 2 Project with 3Skill)

An end-to-end Natural Language Processing (NLP) and Machine Learning project that automatically classifies customer reviews into **Positive**, **Negative**, or **Neutral** sentiments[cite: 1].

## 🔗 Live Demo
Experience the interactive web application here: 
[🔗 AI-Powered Sentiment Analysis App](https://ai-powered-sentiment-analysis-system-mgla3rrovm9atmequvcgcl.streamlit.app/)

---

## 📊 Dataset Overview
The project utilizes a structured customer review dataset (`reviews.csv`) containing:
* **Total Rows**: 10,000 records
* **Columns**: 
  * `review_id`: Unique identifier for each review
  * `review_text`: The textual content of the customer feedback
  * `sentiment`: Target label (**Positive**, **Neutral**, **Negative**)
  * `category`: Business domain or product category
  * `source`: Origin channel of the review

---

## 🛠️ Tech Stack & Architecture
* **Language**: Python
* **Data Processing & Manipulation**: Pandas, NumPy
* **Machine Learning & NLP**: Scikit-learn (TF-IDF Vectorizer, Logistic Regression)
* **Model Validation**: 5-Fold Stratified Cross-Validation, GridSearchCV
* **Visualization**: Matplotlib, Seaborn (`viridis` colormap, bar/violin plots, heatmaps, confusion matrices)
* **Web Framework**: Streamlit
* **Version Control**: Git & GitHub

---

## 🚀 Quick Start & Installation

1. **Clone or download the repository** into your local environment[cite: 1].
2. **Install dependencies**:[cite: 1]
   ```bash
   pip install -r requirements.txt
