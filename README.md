# Twitter Sentiment Analysis with BERT

This project performs sentiment analysis on tweets using both traditional machine learning (Naive Bayes) and deep learning (BERT with TensorFlow). It covers the full pipeline from data preprocessing to model evaluation and visualization.

---

## 🔧 Features

- 📊 Exploratory Data Analysis using `ydata_profiling`
- 🧹 Text preprocessing (regex, stopwords, stemming)
- ⚖️ Class balancing with `RandomOverSampler`
- 🧠 Naive Bayes model using TF-IDF
- 🤖 Fine-tuned BERT model using `TFBertModel` from Hugging Face
- 📈 Accuracy, F1-score, and confusion matrix evaluation
- 📉 Visualizations using Matplotlib and Seaborn

---

## 🛠️ Tech Stack

- Python, NumPy, pandas, seaborn, matplotlib
- Scikit-learn, imbalanced-learn
- Hugging Face Transformers (`BertTokenizerFast`, `TFBertModel`)
- TensorFlow and Keras
- ydata-profiling
