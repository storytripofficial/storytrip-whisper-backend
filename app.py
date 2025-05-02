import whisper
from flask import Flask, request, jsonify
import os

app = Flask(__name__)
model = whisper.load_model("tiny")  # Soporta español y es el más ligero multilingüe

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    file_path = f"temp_{audio_file.filename}"
    audio_file.save(file_path)

    try:
        # Desactiva fp16 para evitar errores en CPU y reducir consumo de memoria
        result = model.transcribe(file_path, fp16=False)
        text = result['text']
    except Exception as e:
        os.remove(file_path)
        return jsonify({"error": str(e)}), 500

    os.remove(file_path)
    return jsonify({"text": text})
