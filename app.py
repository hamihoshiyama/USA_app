from flask import  Flask, request, jsonify, send_from_directory
import openai
from flask_cors import CORS
import joblib
import librosa
import numpy as np
import os
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

# OpenAI APIキーの設定
load_dotenv()
openai.api_key = os.getenv("OPEN_API_KEY")

# 学習済み分類器モデルの読み込み
model = joblib.load('audio_classification_model.pkl')

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    intervals = librosa.effects.split(y, top_db=20)
    features = []
    for start, end in intervals:
        segment = y[start:end]
        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        features.append(mfccs_mean)
    return features

def classify_audio(file_path):
    try:
        features = extract_features(file_path)
        labels = model.predict(features)
        counts = np.bincount(labels)
        return np.argmax(counts)
    except Exception as e:
        print(f"Error classifying audio: {e}")
        raise e

def generate_response(label, text):
    prompt_map = {
        0: "ただ愚痴を聞いてください ",
        1: "共感を示してください",
        2: "的確なアドバイスをください "
    }
    prompt = prompt_map[label] + text
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return response.choices[0].message.content.strip()
@app.route('/')
def home():
    return send_from_directory('', 'index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/process_audio', methods=['POST'])
def process_audio():
    file = request.files['file']
    file_path = 'temp.wav'
    file.save(file_path)
    
    label = classify_audio(file_path)
    
    # テスト用の固定テキスト
    text = "テスト音声データの内容をここに記載"
    
    response = generate_response(label, text)
    
    return jsonify({'response': response})
"""
def send_daily_notification():
    now = datetime.now()
    target_time = now.replace(hour=12, minute=0, second=0, microsecond=0)
    if now > target_time:
        target_time += timedelta(days=1)
    delta = (target_time - now).total_seconds()
    threading.Timer(delta, send_daily_notification).start()
    print("Sending notification: 今日の気分は?")

send_daily_notification()
"""

if __name__ == '__main__':
    app.run(debug=True)
