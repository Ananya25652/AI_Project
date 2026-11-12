import speech_recognition as sr
import librosa
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import shap
import soundfile as sf
import io

# ---------------- Step 1: Speech to Text ----------------
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak something...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print("📝 Transcribed Text:", text)
        return text, audio.get_wav_data()
    except sr.UnknownValueError:
        print("❌ Could not understand audio.")
        return "", None
    except sr.RequestError:
        print("⚠ Could not request results, check internet.")
        return "", None

# ---------------- Step 2: Audio Features ----------------
def emotion_from_audio(y, sr):
    rms = np.mean(librosa.feature.rms(y=y))
    pitches = librosa.yin(y, fmin=80, fmax=400)
    pitch = float(np.nanmean(pitches)) if pitches.size>0 else 0

    if rms > 0.05 and pitch > 180:
        return "Excited/Happy 😃"
    elif rms < 0.02 and pitch < 140:
        return "Sad 😢"
    elif rms > 0.07:
        return "Angry 😠"
    else:
        return "Neutral 😐"

# ---------------- Step 3: Explainable AI ----------------
def explain_with_xai(text):
    print("\n🧩 Generating XAI explanations...")

    # Dummy dataset
    texts = ["I love this", "This is good", "Amazing job", 
             "I hate this", "This is bad", "Terrible day"]
    labels = [1,1,1,0,0,0]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression().fit(X, labels)

    # --- LIME ---
    explainer = LimeTextExplainer(class_names=["Negative","Positive"])
    exp = explainer.explain_instance(text, lambda x: model.predict_proba(vectorizer.transform(x)), num_features=6)
    print("\n🔍 LIME Explanation:")
    for w, score in exp.as_list():
        print(f"  {w}: {score:.3f}")

    # --- SHAP (text) ---
    explainer_shap = shap.Explainer(model.predict_proba, vectorizer.transform(texts))
    shap_values = explainer_shap(vectorizer.transform([text]))
    print("\n🔍 SHAP Values for each feature:")
    print(shap_values.values)

# ---------------- Step 4: Combine Everything ----------------
def analyze():
    text, wav_data = speech_to_text()
    if text and wav_data:
        # Sentiment
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        if scores['compound'] >= 0.05:
            sentiment = "Positive 😊"
        elif scores['compound'] < -0.05:
            sentiment = "Negative 😞"
        elif scores['compound'] <= -0.05:
            sentiment = "Angry"
        else:
            sentiment = "Neutral 😐"

        # Emotion
        y, sr_rate = sf.read(io.BytesIO(wav_data))
        emotion = emotion_from_audio(y, sr_rate)

        print("\n📊 Sentiment:", sentiment)
        print("🎭 Emotion from Voice:", emotion)

        # XAI
        explain_with_xai(text)
    else:
        print("⚠ No valid input captured.")

# ---------------- Run ----------------
if __name__ == "__main__":
    analyze()
