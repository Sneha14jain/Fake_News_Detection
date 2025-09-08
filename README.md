# Fake_News_Detection
# 📰 Fake News Detection using Machine Learning  

This project focuses on building a **Fake News Detection model** that classifies news articles as **REAL** or **FAKE** using **Natural Language Processing (NLP)** and **Machine Learning techniques**.  
The rise of fake news has created challenges in verifying trustworthy information. This model aims to assist in detecting misleading or fabricated news efficiently.  

---

## 📑 Table of Contents
- [🚀 Features](#-features)  
- [🛠️ Tech Stack](#️-tech-stack)  
- [📂 Dataset](#-dataset)  
- [📊 Model Training & Evaluation](#-model-training--evaluation)  
- [📌 Future Improvements](#-future-improvements)  
- [👩‍💻 Author](#-author)  

---

## 🚀 Features  
- Preprocessing of raw text (removing stopwords, punctuation, and special characters).  
- Feature extraction using **TF-IDF Vectorization**.  
- Training a **Passive Aggressive Classifier** for binary classification.  
- Achieved an accuracy of **92.8%** on the dataset.  
- Evaluation using **confusion matrix** and performance metrics.  

---

## 🛠️ Tech Stack  
- **Python 3**  
- **Jupyter Notebook**  

**Libraries:**  
- `pandas`  
- `numpy`  
- `scikit-learn`  
- `nltk`  
- `matplotlib`  

---

## 📂 Dataset  
The dataset used contains labeled news articles categorized as **REAL** or **FAKE**.  

**Columns include:**  
- `title`: The news headline.  
- `text`: The body of the article.  
- `label`: Target variable (`REAL` or `FAKE`).  

---

## 📊 Model Training & Evaluation  
1. Preprocess dataset using **TF-IDF Vectorizer**.  
2. Train a **Passive Aggressive Classifier** on the training set.  
3. Evaluate with accuracy, precision, recall, and F1-score.  
4. Visualize results using a **confusion matrix**.  

✅ Achieved **92.8% accuracy** on test data.  

---

## 📌 Future Improvements  
- Use **deep learning models** (e.g., LSTMs, Transformers like BERT).  
- Deploy the model as a **web application** using Flask/Streamlit.  
- Expand dataset for better generalization.  

---
