import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from datetime import datetime
from PIL import Image
import pytesseract
import re
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
from sqlalchemy.exc import IntegrityError
from whitenoise import WhiteNoise
# Configure Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)
app.wsgi_app = WhiteNoise(app.wsgi_app, root="static/")
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///phishing_detector.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# Add this line to handle static files
app.wsgi_app = WhiteNoise(app.wsgi_app, root="static/") 
app.config['SECRET_KEY'] = 'your-secret-key'

db = SQLAlchemy(app)

# Load models
MODEL_FOLDER = os.path.join(os.getcwd(), 'models')
model_files = {
    'Igbm_url_model': 'Igbm_url_model.pkl',
    'Ir_text_model': 'Ir_text_model.pkl',
    #'nb_text_model': 'nb_text_model.pkl',
    'rf_url_model': 'rf_url_model.pkl',
    'svm_url_model': 'svm_url_model.pkl',
    'xgb_text_model': 'xgb_text_model.pkl',
    'tfidf_vectorizer': 'tfidf_vectorizer.pkl'
}

models = {}
for name, file in model_files.items():
    path = os.path.join(MODEL_FOLDER, file)
    if os.path.exists(path):
        models[name] = joblib.load(path)
    else:
        raise FileNotFoundError(f"Model {name} not found at {path}")

# Database models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class EmailAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    is_phishing = db.Column(db.Boolean, nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    detected_indicators = db.Column(db.Text, nullable=True)
    urls_found = db.Column(db.Text, nullable=True)
    model_predictions = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# Login decorators
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'danger')
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

def login_required_api(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'Login required'}), 401
        return f(*args, **kwargs)
    return decorated_function

# ML functions
def predict_url_models(url):
    features = models['tfidf_vectorizer'].transform([url])
    return {
        'Igbm_url_model': bool(models['Igbm_url_model'].predict(features)[0]),
        'rf_url_model': bool(models['rf_url_model'].predict(features)[0]),
        'svm_url_model': bool(models['svm_url_model'].predict(features)[0])
    }

def predict_text_models(text):
    features = models['tfidf_vectorizer'].transform([text])
    return {
        'Ir_text_model': bool(models['Ir_text_model'].predict(features)[0]),
        #'nb_text_model': bool(models['nb_text_model'].predict(features)[0]),
        'xgb_text_model': bool(models['xgb_text_model'].predict(features)[0])
    }

def analyze_email_content(text):
    indicators = {
        'urgency': {'pattern': r'urgent|immediate|action\s+required|account.*suspend|before\s+it\'s\s+too\s+late', 'weight': 30},
        'personal_info': {'pattern': r'verify.*account|confirm.*password|update.*details|click.*here', 'weight': 25},
        'threats': {'pattern': r'terminate|suspend|close.*account|account\s+locked|unauthorized\s+access', 'weight': 20},
        'poor_grammar': {'pattern': r'you\s+is|they\s+is|we\s+is|have\s+been\s+sent', 'weight': 10},
        'generic_greeting': {'pattern': r'dear\s+user|dear\s+customer|dear\s+sir/madam', 'weight': 10},
        'suspicious_urls': {'pattern': r'(https?://\S+)', 'weight': 5},
    }

    detected_indicators = []
    confidence_score = 0
    urls_found = []

    urls = re.findall(r'(https?://\S+)', text)
    url_preds = []
    for url in urls:
        urls_found.append(url)
        preds = predict_url_models(url)
        url_preds.append({url: preds})
        if any(preds.values()):
            detected_indicators.append('url_ml_phishing')
            confidence_score += 15
        else:
            detected_indicators.append('urls_found')
            confidence_score += indicators['suspicious_urls']['weight']

    # Heuristics
    for indicator, data in indicators.items():
        if indicator != 'suspicious_urls' and re.search(data['pattern'], text, re.I):
            detected_indicators.append(indicator)
            confidence_score += data['weight']

    text_preds = predict_text_models(text)
    if any(text_preds.values()):
        detected_indicators.append('text_ml_phishing')
        confidence_score += 20

    is_phishing = confidence_score > 30
    return is_phishing, confidence_score, detected_indicators, urls_found, url_preds, text_preds

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return render_template('index.html')
    return redirect(url_for('login_page'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')
@app.route('/login', methods=['GET', 'POST'])
def login_page():
    error_username = None
    error_password = None
    username_or_email = ""

    if request.method == 'POST':
        username_or_email = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username_or_email or not password:
            flash("Please enter both username/email and password", "warning")
            return render_template(
                'login.html',
                username_value=username_or_email,
                error_username=None,
                error_password=None
            )

        # Query user by username OR email
        user = User.query.filter(
            (User.username == username_or_email) | (User.email == username_or_email)
        ).first()

        if not user:
            error_username = "🚫 User is not registered."
        elif not user.check_password(password):
            error_password = "❌ Invalid password."
        else:
            # Successful login
            session['user_id'] = user.id
            #flash(f"✅ Welcome back, {user.username}!", "success")
            return redirect(url_for('dashboard'))

    return render_template(
        'login.html',
        username_value=username_or_email,
        error_username=error_username,
        error_password=error_password
    )



@app.route('/register', methods=['GET', 'POST'])
def register_page():
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip()
        password = request.form['password']

        # Create a new user object
        new_user = User(username=username, email=email)
        new_user.set_password(password)

        try:
            db.session.add(new_user)
            db.session.commit()
            flash("✅ Account created successfully! You can now login.", "success")
            return redirect(url_for('login_page'))

        except IntegrityError:
            db.session.rollback()
            # Check if username or email exists
            existing_user = User.query.filter(
                (User.username == username) | (User.email == email)
            ).first()
            if existing_user:
                flash("🚫 Account already exists. If you forgot your password, please reset it.", "error")
            else:
                flash("⚠️ Registration failed. Please try again.", "error")

    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login_page'))


@app.route('/logout_confirm')
@login_required
def logout_confirm():
    # Render a page that will immediately show the SweetAlert popup
    return render_template('logout_confirm.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password_page():
    if request.method == 'POST':
        username_or_email = request.form['username']
        new_password = request.form['new_password']

        # Find user by username OR email
        user = User.query.filter(
            (User.username == username_or_email) | (User.email == username_or_email)
        ).first()

        if user:
            # Reset password
            user.set_password(new_password)
            db.session.commit()

            flash('✅ Your password has been reset. Please log in.', 'success')
            return redirect(url_for('login_page'))
        else:
            flash('❌ User not found. Please check username/email.', 'danger')

    return render_template('forgot_password.html')


@app.route('/upload', methods=['POST'])
@login_required_api
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    try:
        image = Image.open(filepath)
        text = pytesseract.image_to_string(image)
        if len(text.strip()) < 30 or len(re.findall(r'\w+', text)) < 10:
            return jsonify({'error': 'Uploaded image does not appear to contain a valid email.'}), 400
        is_phishing, confidence, indicators, urls_found, url_preds, text_preds = analyze_email_content(text)
        analysis = EmailAnalysis(
            filename=filename,
            is_phishing=is_phishing,
            confidence_score=confidence,
            detected_indicators=','.join(indicators),
            urls_found=','.join(urls_found),
            model_predictions=str({'url_models': url_preds, 'text_models': text_preds}),
            user_id=session['user_id']
        )
        db.session.add(analysis)
        db.session.commit()
        return jsonify({
            'success': True,
            'is_phishing': is_phishing,
            'confidence': confidence,
            'indicators': indicators,
            'urls_found': urls_found,
            'url_model_predictions': url_preds,
            'text_model_predictions': text_preds,
            'text': text
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
@login_required
def history():
    user_id = session['user_id']
    analyses = EmailAnalysis.query.filter_by(user_id=user_id).order_by(EmailAnalysis.timestamp.desc()).all()
    return render_template('history.html', analyses=analyses)

@app.route('/tips')
def tips():
    return render_template('tips.html')

@app.route('/report')
def report():
    return render_template('report.html')
@app.route('/analyze', methods=['POST'])
def analyze_image():
    """
    This route receives the uploaded image, processes it,
    and returns a JSON response.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # --- THIS IS WHERE YOU ADD YOUR ML MODEL LOGIC ---
        # For now, we'll return a dummy result.
        # Replace this with your actual prediction logic.
        
        # is_phishing_result = your_model.predict(file)
        # confidence_result = your_model.get_confidence(file)

        # Dummy data:
        is_phishing_result = True
        confidence_result = 0.95

        # ----------------------------------------------------

        # Return the results in the correct JSON format
        return jsonify({
            'is_phishing': is_phishing_result,
            'confidence_score': confidence_result
        })

    return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
