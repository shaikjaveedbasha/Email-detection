import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import pickle
from tqdm import tqdm

# ------------------------------
# Load CSV and limit to 50k rows
# ------------------------------
df_urls = pd.read_csv("archive/urls.csv")
print("Columns in CSV:", df_urls.columns)

df_urls = df_urls.sample(n=50000, random_state=42)
print("Using", len(df_urls), "rows for training.")

# ------------------------------
# Features and labels
# ------------------------------
X_url = df_urls["url"]
y_url = df_urls["type"]

# Encode labels to integers
label_encoder = LabelEncoder()
y_url_encoded = label_encoder.fit_transform(y_url)

# Save label encoder
with open("url_label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# ------------------------------
# TF-IDF Vectorization with progress bar
# ------------------------------
print("TF-IDF vectorization:")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_url_tfidf = tfidf_vectorizer.fit_transform(tqdm(X_url, total=len(X_url), desc="Vectorizing URLs"))

# Save TF-IDF vectorizer
with open("url_tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)

# ------------------------------
# Train LightGBM
# ------------------------------
print("Training LightGBM model...")
lgb_train = lgb.Dataset(X_url_tfidf, label=y_url_encoded)
lgb_params = {
    "objective": "multiclass",
    "num_class": len(label_encoder.classes_),
    "metric": "multi_logloss",
    "verbosity": -1
}
lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=100)

# Save LightGBM model
lgb_model.save_model("url_lgb_model.txt")

# ------------------------------
# Train Random Forest with progress bar
# ------------------------------
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=1)
rf_model.fit(X_url_tfidf, y_url_encoded)

# Save Random Forest model
with open("url_rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

print("URL models training complete!")
