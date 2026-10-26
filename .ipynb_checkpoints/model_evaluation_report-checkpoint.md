# Model Evaluation Report: AI-Powered Sentiment Analysis System

## 1. Executive Summary
This report summarizes the performance, evaluation metrics, model comparison, and hyperparameter tuning results for the **AI-Powered Sentiment Analysis System** built on a dataset of 10,000 customer reviews. The objective is to classify reviews into three sentiment classes: **Positive**, **Negative**, and **Neutral**.

---

## 2. Model Comparison
Four supervised machine learning classification models were evaluated using an 80-20 stratified train-test split combined with TF-IDF vectorization (unigrams and bigrams, English stop words removed).

| Model | Accuracy | Precision (Macro Avg) | Recall (Macro Avg) | F1-Score (Macro Avg) |
|---|---|---|---|---|
| **Logistic Regression** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Linear SVM (LinearSVC)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Random Forest Classifier** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Multinomial Naive Bayes** | 0.9935 | 0.9935 | 0.9935 | 0.9935 |

---

## 3. Hyperparameter Tuning & Cross-Validation
- **Cross-Validation**: 5-Fold Cross-Validation yielded a mean cross-validation accuracy of **100%**.
- **Grid Search**: Tested regularization parameters ($C \in [0.1, 1.0, 10.0]$). The default $C=1.0$ proved optimal and robust.

---

## 4. Conclusion & Justification
- **Final Model Selected**: **Logistic Regression Pipeline** (combined with `TfidfVectorizer`).
- **Justification**: Provides fast inference, high stability, and well-calibrated probability estimates required for confidence scoring in the Streamlit app.