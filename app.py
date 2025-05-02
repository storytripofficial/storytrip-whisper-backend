import whisper
from flask import Flask, request, jsonify
import os

app = Flask(__name__)
model = whisper.load_model("tiny")  # modelo ligero para Render gratuito

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    file_path = f"temp_{audio_file.filename}"
    audio_file.save(file_path)

    try:
        result = model.transcribe(file_path)
    except Exception as e:
        os.remove(file_path)
        return jsonify({"error": str(e)}), 500

    os.remove(file_path)
    return jsonify({"text": result['text']})
