import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import pickle
from tqdm import tqdm

# ------------------------------
# Load CSV
# ------------------------------
df_text = pd.read_csv("text/text.csv")
print("Columns in CSV:", df_text.columns)

# ------------------------------
# Features and labels
# ------------------------------
# Make sure to use correct column names
X_text = df_text['Email Text'].fillna('')  # replace NaN with empty string
y_text = df_text['Email Type']

# ------------------------------
# Encode labels
# ------------------------------
label_encoder = LabelEncoder()
y_text_encoded = label_encoder.fit_transform(y_text)

# Save label encoder
with open("text_label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# ------------------------------
# TF-IDF Vectorization
# ------------------------------
print(f"TF-IDF vectorization:")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_text_tfidf = tfidf_vectorizer.fit_transform(tqdm(X_text, total=len(X_text)))

# Save TF-IDF vectorizer
with open("text_tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)

# ------------------------------
# Train Logistic Regression
# ------------------------------
print("Training Logistic Regression model...")
logreg_model = LogisticRegression(max_iter=1000, n_jobs=-1)
logreg_model.fit(X_text_tfidf, y_text_encoded)

with open("text_logreg_model.pkl", "wb") as f:
    pickle.dump(logreg_model, f)

# ------------------------------
# Train LightGBM
# ------------------------------
print("Training LightGBM model...")
lgb_train = lgb.Dataset(X_text_tfidf, label=y_text_encoded)
lgb_params = {
    "objective": "multiclass",
    "num_class": len(label_encoder.classes_),
    "metric": "multi_logloss",
    "verbosity": -1
}
lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=100)
lgb_model.save_model("text_lgb_model.txt")

# ------------------------------
# Train Random Forest
# ------------------------------
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf_model.fit(X_text_tfidf, y_text_encoded)

with open("text_rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

print("Text models training complete!")
