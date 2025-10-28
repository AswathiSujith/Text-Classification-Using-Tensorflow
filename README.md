# 🧠 Text Classification using TensorFlow

A **Deep Learning-based Text Classification** project built with **TensorFlow** and **Keras**.  
This project demonstrates how to preprocess text data, build and train an LSTM-based neural network, and evaluate performance using real-world datasets like **IMDB movie reviews** for **sentiment analysis**.

---

## 📘 Table of Contents

1. [Introduction](#introduction)  
2. [Objectives](#objectives)  
3. [System Architecture](#system-architecture)  
4. [Installation](#installation)  
5. [Implementation](#implementation)  
6. [Results](#results)  
7. [Applications](#applications)  
8. [Future Work](#future-work)  
9. [Contributors](#contributors)

---

## 🔍 Introduction

**Text Classification** is a fundamental task in **Natural Language Processing (NLP)** that involves assigning predefined categories to text documents, such as spam detection, sentiment analysis, or topic categorization.  
Using **TensorFlow**, this project applies deep learning models (LSTM, CNN, BERT) to automatically classify text data with high accuracy.

---

## 🎯 Objectives

- Implement text classification using **TensorFlow**  
- Preprocess and clean text data for model training  
- Apply deep learning architectures like **LSTM**, **GRU**, and **CNN**  
- Evaluate performance using **accuracy**, **precision**, **recall**, and **F1-score**

---

## ⚙️ System Architecture

**Workflow:**

Text Input → Preprocessing → Embedding Layer → Deep Learning Model → Classification Output


### Components

1. **Data Collection** – IMDB or custom dataset  
2. **Data Preprocessing** – Tokenization, stopword removal, and padding  
3. **Model Design** – LSTM-based Sequential model using TensorFlow/Keras  
4. **Training & Evaluation** – Train on labeled data and evaluate metrics  
5. **Deployment** – Use the trained model for unseen text classification

---

## 🧩 Installation

### Prerequisites
- Python 3.8 or higher  
- TensorFlow 2.x  
- Jupyter Notebook

### Required Libraries
```bash
pip install tensorflow numpy pandas scikit-learn nltk matplotlib
```

## Implementation

### Sample Model Setup:
```bash
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# Load dataset
max_words = 10000
max_len = 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

# Preprocess data
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# Build model
model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.2)

# Evaluate model
loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")
```

## 📊 Results

Accuracy: ~87% on IMDB sentiment analysis dataset
Observation: LSTM effectively captured sequential dependencies in text
Key Insight: Text preprocessing significantly improved performance

## 🚀 Applications

Spam Filtering: Detect spam vs. legitimate emails
Sentiment Analysis: Analyze opinions from reviews or tweets
Topic Categorization: Classify news articles or blog posts
Chatbots: Intent detection for user interactions

## 🔮 Future Work

Integrate BERT or Transformer models for improved accuracy
Train on multilingual datasets
Deploy as a web app or API for real-time classification

## 👥 Contributors

Author: Aswathi Sujith
