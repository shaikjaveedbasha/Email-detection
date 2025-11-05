import re
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------- URL FEATURES ----------------
def extract_url_features(url):
    features = {}
    features['length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['has_ip'] = 1 if re.match(r'http[s]?://\d+\.\d+\.\d+\.\d+', url) else 0
    features['suspicious_words'] = sum(word in url.lower() for word in ['login', 'verify', 'update', 'secure', 'account'])
    features['num_qmarks'] = url.count('?')
    features['num_equal'] = url.count('=')
    return list(features.values())

def extract_urls(text):
    url_pattern = r'(https?://\S+)'
    return re.findall(url_pattern, text)
