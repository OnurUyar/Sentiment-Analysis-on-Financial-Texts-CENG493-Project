# Sentiment Analysis on Financial Texts (CENG493 Project)
The project I did with my teammates at the end of the 7th semester for CENG493 (Introduction to Natural Language Processing) course.

The goal of the project is to perform sentiment analysis on Turkish financial texts, classifying them as positive or negative using machine learning and deep learning models.

Turkish finance texts used for model training were collected from [this source](https://www.kaggle.com/code/berkaysanc/turkish-sentiment-analysis-lstm-pytorch/notebook) on Kaggle and saved in the 'sentiment_data.csv' file.

## Purpose of the Project

+ Develop a sentiment analysis model for Turkish financial texts.
+ Utilize different embeddings (Word2Vec & BERT) to train models.
+ Provide sentimental insights to complement financial and stock market analysis.

## Libraries used in Python Notebook

+ **Zemberek NLP** (for Turkish text processing)
+ **Pandas & NumPy** (for data manipulation)
+ **NLTK** (for NLP tasks)
+ **TensorFlow & PyTorch** (for machine learning & deep learning)
+ **Matplotlib & Seaborn** (for visualization)

## Steps Followed

### 1. Data Preprocessing
+ Language Identification & Normalization using Zemberek.
+ Tokenization & Lemmatization to extract root forms of words.
+ Stopword, Number, and Punctuation Removal to clean data.

### 2. Embedding Creation
+ **Word2Vec Model:**
  + Words converted into vectors capturing semantic relationships.
  + Parameters set for vector size, window length, n-gram, and frequency.
+ **BERT Embedding:**
  + Used for deeper language understanding.
  + Extracted average embeddings of words in a sentence.

 ### 3. Model Training
+ Word2Vec Embeddings trained with LSTM & RNN models.
+ BERT Embeddings trained with RNN, Balanced RNN, and Logistic Regression models.
+ Commented code explaining key steps.

## Results & Conclusion
+ BERT-based models achieved higher accuracy and F1-score compared to Word2Vec-based models.
+ BERT captured dependencies in complex sentences better, leading to improved sentiment classification.
+ Computational cost of BERT was significantly higher than Word2Vec models.
+ Preprocessing played a crucial role in improving model performance.

This project successfully demonstrates the integration of deep learning, embeddings, and NLP for financial sentiment analysis. The results emphasize the advantages of transformer-based models like BERT in capturing sentiment nuances in financial texts. Future improvements could include hyperparameter tuning, dataset expansion, and real-time sentiment tracking for financial analysis.
