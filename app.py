from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
import pickle
import numpy as np
from datetime import datetime, timedelta
import os
import secrets
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Change this to a random secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///crop_predictor.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Initialize Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load trained model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Warning: model.pkl not found. Please run train_model.py first.")
    model = None

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    reset_token = db.Column(db.String(100))
    reset_token_expiry = db.Column(db.DateTime)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    nitrogen = db.Column(db.Float, nullable=False)
    phosphorus = db.Column(db.Float, nullable=False)
    potassium = db.Column(db.Float, nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    ph = db.Column(db.Float, nullable=False)
    rainfall = db.Column(db.Float, nullable=False)
    predicted_crop = db.Column(db.String(50), nullable=False)
    date_predicted = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Input validation ranges for agricultural parameters
VALID_RANGES = {
    'nitrogen': {'min': 0, 'max': 150, 'error': 'Nitrogen should be between 0-150 kg/ha'},
    'phosphorus': {'min': 0, 'max': 150, 'error': 'Phosphorus should be between 0-150 kg/ha'},
    'potassium': {'min': 0, 'max': 150, 'error': 'Potassium should be between 0-150 kg/ha'},
    'temperature': {'min': -10, 'max': 50, 'error': 'Temperature should be between -10°C and 50°C'},
    'humidity': {'min': 0, 'max': 100, 'error': 'Humidity should be between 0-100%'},
    'ph': {'min': 0, 'max': 14, 'error': 'pH should be between 0-14'},
    'rainfall': {'min': 0, 'max': 3000, 'error': 'Rainfall should be between 0-3000 mm'}
}

# Mapping of crop names to their image files
CROP_IMAGES = {
    'rice': 'rice.jpg',
    'maize': 'maize.jpg',
    'chickpea': 'chickpea.jpg',
    'kidneybeans': 'kidneybeans.jpg',
    'pigeonpeas': 'pigeonpeas.jpg',
    'mothbeans': 'mothbeans.jpg',
    'mungbean': 'mungbean.jpg',
    'blackgram': 'blackgram.jpg',
    'lentil': 'lentil.jpg',
    'pomegranate': 'pomegranate.jpg',
    'banana': 'banana.jpg',
    'mango': 'mango.jpg',
    'grapes': 'grapes.jpg',
    'watermelon': 'watermelon.jpg',
    'muskmelon': 'muskmelon.jpg',
    'apple': 'apple.jpg',
    'orange': 'orange.jpg',
    'papaya': 'papaya.jpg',
    'coconut': 'coconut.jpg',
    'cotton': 'cotton.jpg',
    'jute': 'jute.jpg',
    'coffee': 'coffee.jpg',
}

# Default image if crop image is not found
DEFAULT_CROP_IMAGE = 'default_crop.jpg'

def send_password_reset_email(recipient_email, recipient_name, reset_link):
    """Send password reset email via Resend API (preferred on cloud) or SMTP fallback."""
    subject = "Password Reset Request"
    body = (
        f"Hello {recipient_name},\n\n"
        f"To reset your password, visit the following link:\n{reset_link}\n\n"
        "If you did not make this request, simply ignore this email and no changes will be made."
    )

    # Preferred for deployed environments where SMTP egress is blocked.
    resend_api_key = (os.getenv('RESEND_API_KEY', '') or '').strip()
    if resend_api_key:
        resend_from = (os.getenv('RESEND_FROM', os.getenv('SMTP_FROM', 'onboarding@resend.dev')) or '').strip()
        response = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {resend_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "from": resend_from,
                "to": [recipient_email],
                "subject": subject,
                "text": body,
            },
            timeout=15,
        )
        response.raise_for_status()
        return "resend"

    # SMTP fallback (useful for local/dev).
    smtp_host = (os.getenv('SMTP_HOST', 'smtp.gmail.com') or '').strip()
    smtp_port = int((os.getenv('SMTP_PORT', '587') or '587').strip())
    smtp_username = (os.getenv('SMTP_EMAIL', '') or '').strip()
    # Gmail app password is shown in groups; remove spaces to avoid auth failures.
    smtp_password = (os.getenv('SMTP_APP_PASSWORD', '') or '').replace(' ', '').strip()
    sender_email = (os.getenv('SMTP_FROM', smtp_username) or '').strip()
    smtp_use_ssl = (os.getenv('SMTP_USE_SSL', 'false') or '').strip().lower() == 'true'
    smtp_use_tls = (os.getenv('SMTP_USE_TLS', 'true') or '').strip().lower() == 'true'

    if not smtp_username or not smtp_password:
        raise ValueError("SMTP credentials missing. Set SMTP_EMAIL and SMTP_APP_PASSWORD, or RESEND_API_KEY.")

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    if smtp_use_ssl or smtp_port == 465:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=10) as server:
            server.login(smtp_username, smtp_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
    else:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.ehlo()
            if smtp_use_tls:
                server.starttls()
                server.ehlo()
            server.login(smtp_username, smtp_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
    return "smtp"

def validate_inputs(inputs):
    """Validate if inputs are within realistic agricultural ranges"""
    validation_errors = []
    
    # Check each parameter against its valid range
    if not (VALID_RANGES['nitrogen']['min'] <= inputs['nitrogen'] <= VALID_RANGES['nitrogen']['max']):
        validation_errors.append(VALID_RANGES['nitrogen']['error'])
    
    if not (VALID_RANGES['phosphorus']['min'] <= inputs['phosphorus'] <= VALID_RANGES['phosphorus']['max']):
        validation_errors.append(VALID_RANGES['phosphorus']['error'])
    
    if not (VALID_RANGES['potassium']['min'] <= inputs['potassium'] <= VALID_RANGES['potassium']['max']):
        validation_errors.append(VALID_RANGES['potassium']['error'])
    
    if not (VALID_RANGES['temperature']['min'] <= inputs['temperature'] <= VALID_RANGES['temperature']['max']):
        validation_errors.append(VALID_RANGES['temperature']['error'])
    
    if not (VALID_RANGES['humidity']['min'] <= inputs['humidity'] <= VALID_RANGES['humidity']['max']):
        validation_errors.append(VALID_RANGES['humidity']['error'])
    
    if not (VALID_RANGES['ph']['min'] <= inputs['ph'] <= VALID_RANGES['ph']['max']):
        validation_errors.append(VALID_RANGES['ph']['error'])
    
    if not (VALID_RANGES['rainfall']['min'] <= inputs['rainfall'] <= VALID_RANGES['rainfall']['max']):
        validation_errors.append(VALID_RANGES['rainfall']['error'])
    
    return validation_errors

# Routes
@app.route('/')
def home():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        fullname = request.form.get('fullname')
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password')
        
        # Check if email already exists
        user = User.query.filter(func.lower(User.email) == email).first()
        if user:
            flash('Email already exists.')
            return redirect(url_for('register'))
        
        # Create new user
        new_user = User(
            fullname=fullname,
            email=email,
            password_hash=generate_password_hash(password)
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password')
        
        user = User.query.filter(func.lower(User.email) == email).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password')
    
    return render_template('login.html')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        email = (request.form.get('email') or '').strip().lower()
        if not email:
            flash('Please enter a valid email address.', 'warning')
            return redirect(url_for('forgot_password'))

        try:
            user = User.query.filter(func.lower(User.email) == email).first()

            if user:
                # Generate a secure token
                token = secrets.token_urlsafe(32)
                user.reset_token = token
                user.reset_token_expiry = datetime.utcnow() + timedelta(hours=1)  # Token valid for 1 hour
                db.session.commit()

                # Create reset link
                reset_link = url_for('reset_password', token=token, _external=True)

                try:
                    provider = send_password_reset_email(user.email, user.fullname, reset_link)
                    print(f"\n✅ Password reset email successfully sent to {user.email} via {provider}")
                except Exception as e:
                    print(f"\n❌ Failed to send email: {e}")
                    # Fallback to console print if email fails
                    print("\n" + "=" * 60)
                    print(f"🔗 [FALLBACK] Reset Link: {reset_link}")
                    print("=" * 60 + "\n")
                    if app.debug:
                        flash(f"Email could not be sent. Use this reset link: {reset_link}", 'warning')
            else:
                # Same message for security (don't reveal if email exists)
                print(f"\n⚠️  Password reset attempted for non-existent email: {email}")

            # Uniform response to prevent account enumeration.
            flash('If your email exists in our system, a password reset link has been sent.', 'info')
            return redirect(url_for('forgot_password'))
        except SQLAlchemyError as e:
            db.session.rollback()
            app.logger.error(f"Forgot password database error: {e}")
            flash('Unable to process reset request right now. Please try again.', 'error')
            return redirect(url_for('forgot_password'))
        except Exception as e:
            app.logger.error(f"Forgot password unexpected error: {e}")
            flash('Unable to process reset request right now. Please try again.', 'error')
            return redirect(url_for('forgot_password'))
    
    return render_template('forgot_password.html')

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    try:
        user = User.query.filter_by(reset_token=token).first()
    except SQLAlchemyError as e:
        db.session.rollback()
        app.logger.error(f"Reset password lookup error: {e}")
        flash('Unable to process reset request right now. Please try again.', 'error')
        return redirect(url_for('forgot_password'))
    
    # Check if token exists and is valid
    if not user or not user.reset_token_expiry or user.reset_token_expiry < datetime.utcnow():
        print(f"\n❌ Invalid or expired password reset attempt with token: {token}")
        flash('Invalid or expired password reset link. Please request a new one.')
        return redirect(url_for('forgot_password'))
    
    if request.method == 'POST':
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash("Passwords don't match.")
        else:
            # Update password and clear reset token
            user.password_hash = generate_password_hash(password)
            user.reset_token = None
            user.reset_token_expiry = None
            try:
                db.session.commit()
            except SQLAlchemyError as e:
                db.session.rollback()
                app.logger.error(f"Reset password commit error: {e}")
                flash('Unable to reset password right now. Please try again.', 'error')
                return render_template('reset_password.html', token=token)
            
            print(f"\n✅ Password successfully reset for user: {user.email}")
            flash('Your password has been reset successfully. Please login with your new password.', 'success')
            return redirect(url_for('login'))
    
    return render_template('reset_password.html', token=token)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    try:
        # Get user's prediction history
        predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.date_predicted.desc()).all()
        return render_template('dashboard.html', predictions=predictions, user=current_user)
    except Exception as e:
        app.logger.error(f"Dashboard error: {e}")
        flash('An error occurred while loading your dashboard.')
        return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Get input values from form
        nitrogen = float(request.form.get('N'))
        phosphorus = float(request.form.get('P'))
        potassium = float(request.form.get('K'))
        temperature = float(request.form.get('temperature'))
        humidity = float(request.form.get('humidity'))
        ph = float(request.form.get('ph'))
        rainfall = float(request.form.get('rainfall'))
        
        # Collect inputs for validation
        inputs = {
            'nitrogen': nitrogen,
            'phosphorus': phosphorus,
            'potassium': potassium,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall
        }
        
        # Validate inputs
        validation_errors = validate_inputs(inputs)
        
        if validation_errors:
            for error in validation_errors:
                flash(error)
            return redirect(url_for('home'))
        
        # Create features array for prediction
        features = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
        final_features = [np.array(features)]
        
        # Make prediction
        prediction = model.predict(final_features)
        crop = prediction[0]
        
        # Save prediction to database
        new_prediction = Prediction(
            user_id=current_user.id,
            nitrogen=nitrogen,
            phosphorus=phosphorus,
            potassium=potassium,
            temperature=temperature,
            humidity=humidity,
            ph=ph,
            rainfall=rainfall,
            predicted_crop=crop
        )
        
        db.session.add(new_prediction)
        db.session.commit()
        
        # Console output for prediction (useful for debugging)
        print(f"\n🌱 New crop prediction made by {current_user.fullname}:")
        print(f"   Predicted crop: {crop}")
        print(f"   Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get the image filename for the predicted crop
        crop_image = CROP_IMAGES.get(crop.lower(), DEFAULT_CROP_IMAGE)
        crop_image_path = url_for('static', filename=f'images/{crop_image}')
        
        return render_template('result.html', 
                              prediction_text=f'Recommended Crop: {crop}',
                              crop_image=crop_image_path)
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        flash('An error occurred during prediction.')
        return redirect(url_for('home'))

# Development helper route (remove in production)
@app.route('/dev-users')
def dev_users():
    """Development route to see all users - remove in production"""
    if app.debug:
        users = User.query.all()
        print("\n" + "="*40)
        print("👥 REGISTERED USERS:")
        print("="*40)
        for user in users:
            print(f"   Name: {user.fullname}")
            print(f"   Email: {user.email}")
            print(f"   ID: {user.id}")
            print("-" * 30)
        print("="*40 + "\n")
        return f"Found {len(users)} users. Check console for details."
    else:
        return "Not available in production mode"

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create database tables
        print("\n🚀 Smart Crop Predictor - College Project")
        print("📊 Database initialized successfully")
        print("🔗 Access the application at: http://127.0.0.1:5000")
        print("💡 Password reset links will appear in this console\n")
    app.run(debug=True)
