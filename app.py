# Model updated: 2025-11-15 - Now supports 4 risk classes
import sqlite3
import contextlib
import re
import numpy as np
import pickle
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from flask import (
    Flask, render_template, 
    request, session, redirect
)
from twilio.rest import Client
import os

# Fix: Use non-GUI backend for matplotlib to avoid RuntimeError
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from create_database import setup_database
from utils import login_required, set_session

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.secret_key = 'xpSm7p5bgJY8rNoBjGWiz5yjxM-NEBlW6SIBI62OkLc='

model = pickle.load(open('model.pkl', 'rb'))

database = "users.db"
setup_database(name=database)

def send_sms(prediction, to_phone_numbers):
    # Load credentials from environment to avoid committing secrets
    account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
    auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
    from_phone = os.environ.get('TWILIO_FROM_PHONE')

    # If Twilio creds are not configured, skip sending SMS
    if not account_sid or not auth_token or not from_phone:
        print("Twilio credentials not configured. Skipping SMS send.")
        return

    if prediction == 0:
        message_body = "Alert! Your area has high risk landslide possibility. Your place is not safe."
    elif prediction == 1:
        message_body = "Alert! Your area has moderate risk landslide possibility. Your place is not safe."
    elif prediction == 2:
        message_body = "Alert! Your area has low risk landslide possibility. Your place is safe."
    else:
        message_body = "Good news! The area is safe."

    client = Client(account_sid, auth_token)

    for to_phone in to_phone_numbers:
        try:
            message = client.messages.create(
                body=message_body,
                from_=from_phone,
                to=to_phone
            )
            print(f"SMS sent successfully to {to_phone}.")
        except Exception as e:
            print(f"Failed to send SMS to {to_phone}: {e}")

@app.route('/')
def first():
    return render_template("index1.html")

@app.route('/logout')
def logout():
    session.clear()
    session.permanent = False
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    username = request.form.get('username')
    password = request.form.get('password')

    query = 'select username, password, email from users where username = :username'

    with contextlib.closing(sqlite3.connect(database)) as conn:
        with conn:
            account = conn.execute(query, {'username': username}).fetchone()

    if not account:
        return render_template('login.html', error='Username does not exist')

    try:
        ph = PasswordHasher()
        ph.verify(account[1], password)
    except VerifyMismatchError:
        return render_template('login.html', error='Incorrect password')

    if ph.check_needs_rehash(account[1]):
        query = 'update users set password = :password where username = :username'
        params = {'password': ph.hash(password), 'username': account[0]}

        with contextlib.closing(sqlite3.connect(database)) as conn:
            with conn:
                conn.execute(query, params)

    set_session(username=account[0], email=account[2], remember_me='remember-me' in request.form)
    return redirect('/predict1')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')

    password = request.form.get('password')
    confirm_password = request.form.get('confirm-password')
    username = request.form.get('username')
    email = request.form.get('email')

    if len(password) < 8:
        return render_template('register.html', error='Your password must be 8 or more characters')
    if password != confirm_password:
        return render_template('register.html', error='Passwords do not match')
    if not re.match(r'^[a-zA-Z0-9]+$', username):
        return render_template('register.html', error='Username must only be letters and numbers')
    if not 3 < len(username) < 26:
        return render_template('register.html', error='Username must be between 4 and 25 characters')

    query = 'select username from users where username = :username;'
    with contextlib.closing(sqlite3.connect(database)) as conn:
        with conn:
            result = conn.execute(query, {'username': username}).fetchone()

    if result:
        return render_template('register.html', error='Username already exists')

    pw = PasswordHasher()
    hashed_password = pw.hash(password)

    query = 'insert into users(username, password, email) values (:username, :password, :email);'
    params = {'username': username, 'password': hashed_password, 'email': email}

    with contextlib.closing(sqlite3.connect(database)) as conn:
        with conn:
            conn.execute(query, params)

    set_session(username=username, email=email)
    return redirect('/')

@app.route('/graph')
def graph():
    return render_template("graph.html")

@app.route('/feed back')
def feed_back():
    return render_template("feed back.html")

@app.route('/predict1', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        float_features = [float(x) for x in request.form.values()]
        final = [np.array(float_features)]

        if len(final[0]) == 0:
            return render_template('result.html', prediction='No input provided')

        prediction = model.predict(final)[0]
        confidence = model.predict_proba(final)[0]
        confidence_level = confidence.max()
        confidence_class = confidence.argmax()

        rainfall_value = float_features[0]

        to_phone_numbers = [
            '+919361583902',
            '+917736932683'
        ]

        send_sms(prediction, to_phone_numbers)

        # Create graph
        labels = [f'Class {i}' for i in range(len(confidence))]
        values = confidence.tolist()

        plt.figure(figsize=(8, 5))
        # Updated to match theme palette: High (red), Moderate (yellow), Low (blue), Safe (green)
        plt.bar(labels, values, color=['#D72638', '#FFCC00', '#00AEEF', '#2E8B57'])
        plt.xlabel('Risk Classes')
        plt.ylabel('Confidence Level')
        plt.title('Prediction Confidence Levels')
        plt.ylim(0, 1)
        plt.tight_layout()

        # Save to static folder with absolute path
        import os
        static_folder = os.path.join(os.path.dirname(__file__), 'static')
        graph_filename = 'confidence_graph.png'
        graph_path = os.path.join(static_folder, graph_filename)
        plt.savefig(graph_path, dpi=100, bbox_inches='tight')
        plt.close()

        return render_template('result.html', prediction=prediction, confidence=confidence_level*100, graph_filename=graph_filename)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

