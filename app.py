from flask import Flask, request, jsonify
import whisper
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import openai
import os
from dotenv import load_dotenv

app=Flask(__name__)
try:
    whisper_model = whisper.load_model("small")
except Exception as e:
    print(f"Error loading Whisper model: {e}")

#保存されている音声分類モデルのロード
audio_model=joblib.load("audio_classification_model_3.pkl")

#保存されているテキスト分類モデルのロード
text_model=joblib.load("text_classification_model_3.pkl")
vectonizer=joblib.load("text_vectorizer_3.pkl")

sc= StandardScaler()

# 環境変数を読み込む
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# メインページのルート
@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Voice Classification</title>
    </head>
    <body>
        <h1>Voice Classification API</h1>
        <button id="startButton">Start Recording</button>
        <button id="stopButton" disabled>Stop Recording</button>
        <p id="result"></p>

        <script>
            let mediaRecorder;
            let audioChunks = [];

            document.getElementById("startButton").onclick = async () => {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.start();
                document.getElementById("startButton").disabled = true;
                document.getElementById("stopButton").disabled = false;

                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
                    audioChunks = [];
                    const formData = new FormData();
                    formData.append('audio', audioBlob, 'recording.wav');
                    const response = await fetch("/upload", {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();
                    document.getElementById("result").innerText = result.message;
                    document.getElementById("startButton").disabled = false;
                };
            };

            document.getElementById("stopButton").onclick = () => {
                mediaRecorder.stop();
                document.getElementById("stopButton").disabled = true;
            };
        </script>
    </body>
    </html>
    """
# 録音ファイルのアップロード
@app.route('/upload', methods=['POST'])
def upload_file():
    if "audio" not in request.files:
        return jsonify({"message": "No audio file"}), 400
    
    audio_file=request.files["audio"]
    audio_path="./recording.wav"
    audio_file.save(audio_path)

    transcription_result=whisper_model.transcribe(audio_path, fp16=False, language="en")
    text=transcription_result["text"]

    #音声分類
    y, sr= librosa.load(audio_path, sr=16000)
    # 音声データの長さを2秒に固定
    max_length = 2 * sr
    if len(y) < max_length:
        y = np.pad(y, (0, max_length - len(y)), mode='constant')
    else:
        y = y[:max_length]
    
    # MFCCの数を訓練時と一致させる
    n_mfcc = 13
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # 必要に応じてパディングまたはトリミングを行う
    expected_length = 300  # これは訓練時に使用したMFCC行列の時間方向の長さに一致させる
    if mfcc.shape[1] < expected_length:
        mfcc = np.pad(mfcc, ((0, 0), (0, expected_length - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :expected_length]
    
    mfcc_flatten = mfcc.flatten().reshape(1, -1)  # MFCCをフラット化して1次元配列に変換
    mfcc_std = sc.fit_transform(mfcc_flatten)  # 特徴量を標準化
    audio_prediction = audio_model.predict(mfcc_std)

    #テキスト分類
    text_vectorized=vectonizer.transform([text]).toarray()
    text_prediction=text_model.predict(text_vectorized)

    if audio_prediction[0] == text_prediction[0]:
        final_label=audio_prediction[0]
    else:
        final_label=text_prediction[0]

    prompt_map = {
        0: "Please respond professionally and amicably to people who express fear, apprehension, and anxiety following the shown instructions. Validate their feelings by letting them know it's okay not to be okay. Don't tell them to calm down. Encourage them to focus on things they can change. ",
        1: "Please respond professionally to people who express anger, annoynance, and irritation by following the shown instructions. Quell daily worries, anxieties, work, and evaluations from others that may pop into the head, and encourage to focus only on the “now”. Stay calm and try not to lash out in response, even if it's difficult. Give the person space to self-regulate. Practice active listening",
        2: "Please respond professionally to people who ecpress boredness and disgust. Ask open-ended questions to stimulate their creativity in finding interesting solutions for alleviating boredom. Help them find an engaging activity or one you can participate in together. ",
        3: "Please respond profesiionally to people who express sadness and depression by following the shown instructions. Show empathy. Explain that depression is a complex condition, not a personal flaw or weakness. Ask questions to get more information instead of assuming you understand what they mean. Validate their feelings. You might say, 'That sounds really difficult. I'm sorry to hear that. '"
    }
    prompt = prompt_map.get(final_label, "Be a normal assistant when someone asks for advice")
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a professional mental trainer."+ prompt},
            {"role": "user", "content": text}
        ],
        max_tokens=150
    )

    result_message=response.choices[0].message.content.strip()
    return jsonify({"message":result_message})

if __name__ == "__main__":
    app.run(debug=True)




