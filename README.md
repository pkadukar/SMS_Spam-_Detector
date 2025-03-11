# Spam Classifier

## Project Overview
This project is a spam classifier that determines whether a given email or SMS message is spam or not (ham). It utilizes natural language processing (NLP) techniques and machine learning algorithms to accurately classify messages based on their text content.

## Tech Stack & Libraries Used
- Python
- Pandas
- NumPy
- Matplotlib & Seaborn (for data visualization)
- NLTK (Natural Language Toolkit)
- Sklearn (Scikit-learn for machine learning models)
- XGBoost (Extreme Gradient Boosting for classification)

## Dataset Information
- **Source**: The dataset is obtained from the UCI Machine Learning Repository.
- **Description**: It contains labeled SMS messages as either "spam" or "ham" (not spam).
- **Features Extracted**:
  - Number of characters
  - Number of words
  - Number of sentences
  - Presence of specific keywords commonly found in spam messages

## Text Preprocessing & Feature Engineering
- **Tokenization**: Splitting text into individual words.
- **Stopword Removal**: Eliminating common words that do not contribute much meaning.
- **Stemming**: Using Porter Stemmer to reduce words to their root forms.
- **Vectorization**:
  - Bag-of-Words (BoW)
  - Term Frequency-Inverse Document Frequency (TF-IDF)

## Model Selection & Evaluation
Multiple machine learning models were tested, including:
- Naïve Bayes
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- XGBoost

**Performance Metrics**:
- Accuracy
- Precision
- Recall
- F1-score

The model with the best performance was selected for the final classification task.

## Installation & Usage
### Prerequisites
This project was developed using Jupyter Notebook
pip install pandas numpy matplotlib seaborn nltk scikit-learn xgboost
```

### Running the Classifier
1. Clone the repository:
   ```sh
   git clone <repo_url>
   cd spam-classifier
   ```
2. Run the script to train and test the model:
   ```sh
   python spam_classifier.py
   ```
3. Provide input text messages for classification and get predictions.

## Future Improvements
- Implementing deep learning models like LSTMs for better accuracy.
- Hyperparameter tuning to optimize model performance.
- Expanding the dataset to include more diverse spam messages.

