# Fake_Job_Posting_Detection
Fake job detection using Bernoulli &amp; Exact Bayes with SMOTE and threshold dialer

# 🕵️‍♀️ Fake Job Posting Detection Using Bernoulli & Exact Bayes

This project focuses on identifying fraudulent job postings using supervised machine learning models. Trained on over **500,000+ listings**, it compares the performance of **Bernoulli Naive Bayes** and a **custom Exact Bayes** classifier, enhanced with **SMOTE** and a **threshold-based dialer system** to optimize classification.

---

## 📌 Project Objectives

- Detect **fake job listings** in real-world job portals
- Address **class imbalance** with SMOTE oversampling
- Optimize fraud classification through a **threshold-tuning dialer**
- Compare model performances using key evaluation metrics

---

## 📁 Dataset

- Source: [Kaggle – Real vs. Fake Job Postings](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- Size: 500K+ listings
- Features: `title`, `company_profile`, `description`, `location`, `requirements`, `salary_range`, `telecommuting`, `has_company_logo`, etc.

---

## 🧠 Models Implemented

| Model               | Accuracy | F1 Score |
|--------------------|----------|----------|
| Bernoulli Naive Bayes (SMOTE) | 91%      | 0.93     |
| Exact Bayes (SMOTE)           | 94%      | 0.96     |

- **BernoulliNB** was trained on sparse binary text data + numeric features.
- **Exact Bayes** is a fully custom implementation of the Bayes theorem from scratch using log probabilities and smoothing.

---

## 🎯 Techniques Used

- **Text Vectorization**: CountVectorizer (max_features=5000)
- **Imbalanced Handling**: SMOTE
- **Custom Dialer**: Adjusts classification threshold to fine-tune precision/recall
- **Laplace Smoothing**: Avoids zero probabilities in Exact Bayes
- **Evaluation**: Accuracy, Precision, Recall, F1-score, Confusion Matrix

---

## 🧪 Output
Bernoulli Naive Bayes:
Accuracy: 0.91
F1-Score: 0.93

Exact Bayes:
Accuracy: 0.94
F1-Score: 0.96


Custom Dialer Example (Threshold = 0.5,0.7):
For Exact Bayes
<img width="458" alt="image" src="https://github.com/user-attachments/assets/521e1f43-bbde-4a03-93cf-8e66a47b0b8e" />

For Naive Bayes:(Threshold = 0.01,0.02):
<img width="348" alt="image" src="https://github.com/user-attachments/assets/9f353e32-975c-4920-8bcd-70961c9921a6" />


Confusion Matrix:
[[4558 547]
[ 419 4685]]

## 📄 Full Code & Results (Google Drive Link)
[📥 View Code on Google Drive]([https://drive.google.com/file/d/your_file_id/view?usp=sharing](https://drive.google.com/file/d/1muOEgrDiGGqNVEaSURr-_Y5hUyB49zKf/view?usp=sharing))



