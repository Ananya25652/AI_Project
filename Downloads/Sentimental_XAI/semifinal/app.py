import os
import uuid
import sqlite3
from functools import wraps
from flask import (
    Flask, request, jsonify, render_template, redirect, url_for,
    session, send_from_directory, flash
)
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# audio & ML
from pydub import AudioSegment
import speech_recognition as sr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import librosa, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# XAI
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import shap

# ---------------- App setup ----------------
app = Flask(__name__, template_folder="templates", static_folder="templates")
CORS(app)
app.secret_key = "change_this_to_a_random_secret_in_prod"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DB_PATH = "users.db"
analyzer = SentimentIntensityAnalyzer()

# ---------------- DB helpers ----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()

def create_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (username, generate_password_hash(password)))
        conn.commit()
        return True, None
    except sqlite3.IntegrityError as e:
        return False, "Username already exists"
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return False
    return check_password_hash(row[0], password)

init_db()

# ---------------- auth decorator ----------------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return decorated

# ---------------- Helpers: audio & XAI ----------------
def convert_to_wav(src_path, target_wav_path):
    try:
        audio_segment = AudioSegment.from_file(src_path)
        audio_segment.export(target_wav_path, format="wav")
        return True, None
    except Exception as e:
        return False, str(e)

def sentiment_analysis(text):
    scores = analyzer.polarity_scores(text or "")
    compound = scores.get("compound", 0.0)
    sentiment = "Positive 😊" if compound >= 0.05 else "Negative 😞" if compound <= -0.05 else "Neutral 😐"
    return sentiment, compound, scores

def explain_text_lime(text):
    # lightweight local dummy classifier used for explanation
    texts = ["I love this", "Good job", "Amazing", "I hate this", "couldn't","Bad", "Terrible","not","shouldn't","won't","doesn't","don't","cannot"]
    labels = [1,1,1,0,0,0,0,0,0,0,0,0,0]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression().fit(X, labels)
    explainer = LimeTextExplainer(class_names=["Negative","Positive"])
    exp = explainer.explain_instance(text, lambda x: model.predict_proba(vectorizer.transform(x)), num_features=10)
    # return html badges
    html = ""
    for word, weight in exp.as_list():
        color = "rgba(0,255,0,0.35)" if weight > 0 else "rgba(255,0,0,0.35)"
        html += f"<span style='background:{color};padding:3px 6px;border-radius:5px;margin:3px;display:inline-block'>{word}</span>"
    if not html:
        html = "<span style='color:#888'>No strong LIME features.</span>"
    return html

def explain_audio_shap(feats):
    # feats: 2D numpy array (1 x n_features)
    class DummyAudioModel:
        def predict_proba(self, X):
            X = np.asarray(X, dtype=float)
            X = np.nan_to_num(X)
            pos = np.clip((X.mean(axis=1) + 0.1), 0, 1)
            return np.vstack([1-pos, pos]).T

    model = DummyAudioModel()
    # small constant background
    background = np.zeros((5, feats.shape[1]))
    try:
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_vals = explainer.shap_values(feats, nsamples=30)
    except Exception as e:
        return None, str(e)

    if isinstance(shap_vals, list):
        vals = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
    else:
        vals = shap_vals[0]

    vals = np.array(vals).flatten()
    feat_names = ["RMS", "Centroid", "Rolloff", "ZCR"]
    # truncate or pad
    if len(vals) > len(feat_names):
        vals = vals[:len(feat_names)]
    elif len(vals) < len(feat_names):
        vals = np.pad(vals, (0, len(feat_names)-len(vals)), 'constant')

    # plot
    try:
        plt.figure(figsize=(6,3))
        colors = ["green" if float(v) > 0 else "red" for v in vals]
        bars = plt.bar(range(len(vals)), vals, color=colors)
        plt.xticks(range(len(feat_names)), feat_names)
        plt.title("SHAP Explanation")
        for i, b in enumerate(bars):
            y_pos = float(vals[i])
            label_y = y_pos + (0.001 if y_pos >= 0 else -0.02)
            plt.text(b.get_x() + b.get_width()/2, label_y, f"{vals[i]:.2f}", ha='center', va='bottom', fontsize=8, color='white')
        img_filename = f"shap_{uuid.uuid4().hex}.png"
        img_path = os.path.join(UPLOAD_FOLDER, img_filename)
        plt.tight_layout()
        plt.savefig(img_path, dpi=150)
        plt.close()
        return img_path, None
    except Exception as e:
        return None, str(e)

# ---------------- Routes: pages ----------------
@app.route("/")
def welcome():
    return render_template("welcome.html")

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username","").strip()
        password = request.form.get("password","").strip()
        if not username or not password:
            flash("Username and password required", "error")
            return redirect(url_for("register"))
        ok, err = create_user(username, password)
        if not ok:
            flash(err or "Registration failed", "error")
            return redirect(url_for("register"))
        flash("Registration successful. Please login.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username","").strip()
        password = request.form.get("password","").strip()
        if verify_user(username, password):
            session["username"] = username
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password", "error")
            return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("welcome"))

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", username=session.get("username"))

@app.route("/voice")
@login_required
def voice_page():
    return render_template("voice.html")

@app.route("/text")
@login_required
def text_page():
    return render_template("text.html")

# ---------------- API endpoints for analysis ----------------
@app.route("/api/analyze_audio", methods=["POST"])
@login_required
def api_analyze_audio():
    uploaded = request.files.get("audio")
    if not uploaded:
        return jsonify({"error":"No audio file uploaded"}), 400

    # save and convert
    uid = uuid.uuid4().hex
    raw_path = os.path.join(UPLOAD_FOLDER, f"{uid}_{uploaded.filename}")
    uploaded.save(raw_path)
    wav_path = os.path.join(UPLOAD_FOLDER, f"{uid}.wav")
    ok, err = convert_to_wav(raw_path, wav_path)
    if not ok:
        return jsonify({"error":"Conversion failed", "detail": err}), 500

    # STT
    recognizer = sr.Recognizer()
    transcript = ""
    try:
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            transcript = recognizer.recognize_google(audio_data)
    except Exception:
        transcript = ""

    sentiment, compound, scores = sentiment_analysis(transcript)
    lime_html = explain_text_lime(transcript)

    # audio SHAP
    try:
        y, sr_rate = librosa.load(wav_path, sr=None)
        feats = np.array([
            np.nan_to_num(np.mean(librosa.feature.rms(y=y))),
            np.nan_to_num(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr_rate))),
            np.nan_to_num(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr_rate))),
            np.nan_to_num(np.mean(librosa.feature.zero_crossing_rate(y=y)))
        ]).reshape(1, -1)
        shap_img_path, shap_err = explain_audio_shap(feats)
        shap_url = url_for('uploaded_file', filename=os.path.basename(shap_img_path)) if shap_img_path else None
    except Exception as e:
        shap_url = None
        shap_err = str(e)

    return jsonify({
        "transcript": transcript,
        "sentiment": sentiment,
        "compound": compound,
        "scores": scores,
        "lime_html": lime_html,
        "shap_image": shap_url,
        "shap_error": shap_err if shap_img_path is None else None
    })

@app.route("/api/analyze_text", methods=["POST"])
@login_required
def api_analyze_text():
    data = request.get_json() or {}
    text = data.get("text", "")
    sentiment, compound, scores = None, 0.0, None
    if text:
        sentiment, compound, scores = sentiment_analysis(text)
    else:
        return jsonify({"error":"No text provided"}), 400

    lime_html = explain_text_lime(text)

    # For text SHAP we generate simple numeric features (demo)
    feats = np.array([[len(text), sum(1 for c in text if c.lower() in "aeiou"), text.count(" "), len(text.split())]])
    try:
        shap_img_path, shap_err = explain_audio_shap(feats)
        shap_url = url_for('uploaded_file', filename=os.path.basename(shap_img_path)) if shap_img_path else None
    except Exception as e:
        shap_url = None
        shap_err = str(e)

    return jsonify({
        "sentiment": sentiment,
        "compound": compound,
        "lime_html": lime_html,
        "shap_image": shap_url,
        "shap_error": shap_err if shap_img_path is None else None
    })

# serve uploads
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ---------------- Run ----------------
if __name__ == "__main__":
    init_db()
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
