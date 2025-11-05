# ML models for URL and text detection
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import tensorflow as tf

# ---------- URL MODELS ----------
lgbm_model = lgb.LGBMClassifier()
rf_model = RandomForestClassifier()
svm_model = SVC(probability=True)

def train_url_models(X, y):
    lgbm_model.fit(X, y)
    rf_model.fit(X, y)
    svm_model.fit(X, y)

def ensemble_url_prediction(features):
    probs = np.array([
        lgbm_model.predict_proba([features])[:,1],
        rf_model.predict_proba([features])[:,1],
        svm_model.predict_proba([features])[:,1]
    ])
    avg_prob = np.mean(probs)
    return avg_prob, avg_prob > 0.5

# ---------- TEXT MODELS ----------
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
lr_model = LogisticRegression()
nb_model = MultinomialNB()
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
distilbert_model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

def train_text_models(X_text, y):
    X_tfidf = tfidf_vectorizer.fit_transform(X_text)
    lr_model.fit(X_tfidf, y)
    nb_model.fit(X_tfidf, y)
    xgb_model.fit(X_tfidf, y)
    # DistilBERT fine-tuning is optional; for now, use pre-trained

def ensemble_text_prediction(text):
    # TF-IDF predictions
    X_tfidf = tfidf_vectorizer.transform([text])
    lr_prob = lr_model.predict_proba(X_tfidf)[:,1]
    nb_prob = nb_model.predict_proba(X_tfidf)[:,1]
    xgb_prob = xgb_model.predict_proba(X_tfidf)[:,1]

    # DistilBERT prediction
    tokens = tokenizer(text, return_tensors='tf', truncation=True, padding=True)
    distilbert_prob = tf.nn.softmax(distilbert_model(tokens)['logits'], axis=-1).numpy()[0,1]

    final_prob = 0.4*distilbert_prob + 0.2*lr_prob + 0.2*nb_prob + 0.2*xgb_prob
    return final_prob, final_prob > 0.5
