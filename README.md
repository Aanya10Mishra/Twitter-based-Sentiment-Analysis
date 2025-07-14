# Twitter Sentiment Analysis using Machine Learning

This project performs sentiment analysis on Twitter data using a Machine Learning pipeline. It classifies tweets as **Positive** or **Negative** based on their content. The model is trained using the Sentiment140 dataset and implemented using logistic regression and TF-IDF vectorization.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Workflow](#workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

---

## 📖 Overview

- **Objective**: Automatically detect the sentiment of a tweet (positive or negative).
- **Algorithm Used**: Logistic Regression
- **Vectorization**: TF-IDF (Term Frequency - Inverse Document Frequency)
- **Dataset Size**: 1.6 million tweets
- **Language**: Python

---

## 📂 Dataset

- **Source**: [Sentiment140 - Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Format**: CSV
- **Fields Used**:
  - `text` – the tweet text
  - `target` – sentiment label (0 = Negative, 4 = Positive → converted to 1)

---

## 🧰 Tech Stack

- Python
- Pandas & NumPy
- NLTK (stopword removal & stemming)
- Scikit-learn (TF-IDF, model training)
- Pickle (model serialization)

---

## 🔁 Workflow

1. **Data Collection** – Download and extract dataset using Kaggle API.
2. **Data Cleaning** – Remove noise, special characters, stopwords, and apply stemming.
3. **Label Preprocessing** – Convert `4` (positive) to `1`, making it a binary classification problem.
4. **Vectorization** – Convert textual data into numerical vectors using TF-IDF.
5. **Model Training** – Train Logistic Regression model on 80% of the data.
6. **Evaluation** – Check accuracy on both training and testing data.
7. **Model Saving** – Save the trained model using `pickle`.
8. **Prediction** – Load the saved model to make predictions on new tweet inputs.

---

## 🛠 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
2.Install dependencies:

`pip install -r requirements.txt`

3.Set up Kaggle API key:
- Download your kaggle.json file from your Kaggle account.
- Place it in the project directory.
- Run the following commands:
  ```bash
  mkdir -p ~/.kaggle
  cp kaggle.json ~/.kaggle/
  chmod 600 ~/.kaggle/kaggle.json


## 📊 Results

- The model was trained using Logistic Regression on the Sentiment140 dataset.
- It achieved an accuracy of approximately **82%** on the training data.
- The test data accuracy was also around **82%**, indicating good generalization.
- Preprocessing steps like text cleaning, stemming, and stopword removal helped improve model performance.
- TF-IDF vectorization effectively converted tweets into numerical features suitable for classification.

---

## 🚀 Future Work

- **Add Neutral Sentiment Classification**: Currently, the model only handles positive and negative sentiments. Including neutral class would make it more comprehensive.
- **Implement Advanced NLP Models**: Using LSTM, BERT, or other transformer-based models can improve contextual understanding and accuracy.
- **Develop a Web Interface**: Build a real-time sentiment analysis tool using Streamlit or Flask for easier user interaction.
- **Visualize Results**: Incorporate visual tools like word clouds and sentiment distribution charts using Matplotlib or Plotly.
- **Enable Real-Time Twitter Streaming**: Connect with the Twitter API to analyze live tweets based on hashtags or keywords.
- **Model Deployment**: Package and deploy the model as a REST API or host it on platforms like Hugging Face Spaces, Heroku, or Vercel.

