import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import tensorflow as tf

# -----------------------
# Sample text dataset
# -----------------------
# Columns: 'text', 'label' (0 = safe, 1 = phishing)
data = pd.DataFrame({
    'text': [
        'Your account has been suspended, click here to reactivate.',
        'Hello, your order has been shipped.',
        'Verify your password immediately!',
        'Meeting tomorrow at 10AM.'
    ],
    'label': [1, 0, 1, 0]
})

X_text = data['text'].tolist()
y = data['label'].values

# -----------------------
# TF-IDF + classical ML
# -----------------------
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(X_text)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_tfidf, y)
joblib.dump(lr_model, 'models/lr_text_model.pkl')

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_tfidf, y)
joblib.dump(nb_model, 'models/nb_text_model.pkl')

# XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_tfidf, y)
joblib.dump(xgb_model, 'models/xgb_text_model.pkl')

# -----------------------
# DistilBERT
# -----------------------
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
distilbert_model = TFDistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=2
)
# Optional: fine-tuning on your dataset
# For now, we just save the pre-trained model
distilbert_model.save_pretrained('models/distilbert_text_model')
tokenizer.save_pretrained('models/distilbert_text_model')

print("✅ Text models trained and saved successfully!")
