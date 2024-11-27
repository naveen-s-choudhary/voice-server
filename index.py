import os
import io
import time
import torch
import logging
import magic
from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import nltk
current_dir = os.path.abspath(os.path.dirname(__file__))
nltk_data_path = os.path.join(current_dir, 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Check if the required NLTK data exists, if not, download it to the nltk_data folder
if not os.path.exists(os.path.join(nltk_data_path, 'taggers/averaged_perceptron_tagger')):
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)

# For the 'averaged_perceptron_tagger_eng', handle it similarly
if not os.path.exists(os.path.join(nltk_data_path, 'taggers/averaged_perceptron_tagger_eng')):
    nltk.download('averaged_perceptron_tagger_eng', download_dir=nltk_data_path)



from melo.api import TTS
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"
ckpt_base = 'checkpoints_v2/base_speakers/ses'
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# Initialize ToneColorConverter
tone_color_converter = ToneColorConverter('checkpoints_v2/converter/config.json', device=device)
tone_color_converter.load_ckpt('checkpoints_v2/converter/checkpoint.pth')

# Load base speakers
base_speakers = ['en-au', 'en-br', 'en-default', 'en-india', 'en-newest', 'en-us', 'es', 'fr', 'jp', 'kr', 'zh']
if device == "cpu":
    base_speakers = ['en-br']  # Load fewer models on CPU

key_map = {
    'en-newest': ('EN-Newest', 'EN_NEWEST'),
    'en-us': ('EN-US', 'EN'),
    'en-br': ('EN-BR', 'EN'),
    'en-india': ('EN_INDIA', 'EN'),
    'en-au': ('EN-AU', 'EN'),
    'en-default': ('EN-Default', 'EN'),
    'es': ('ES', 'ES'),
    'fr': ('FR', 'FR'),
    'jp': ('JP', 'JP'),
    'kr': ('KR', 'KR'),
    'zh': ('ZH', 'ZH')
}

source_se = {
    accent: torch.load(f'{ckpt_base}/{accent}.pth').to(device) for accent in base_speakers
}

logging.info('Loaded base speakers.')

model = {}
for accent in base_speakers:
    logging.info(f'Loading TTS model for {accent}...')
    model[accent] = TTS(language=key_map[accent][1], device=device)
logging.info('Loaded TTS models.')

# Helper functions
def validate_audio_file(file):
    allowed_extensions = {'wav', 'mp3', 'flac', 'ogg'}
    max_file_size = 5 * 1024 * 1024  # 5MB

    if not file.filename.split('.')[-1] in allowed_extensions:
        return {"error": "Invalid file type. Allowed types are: wav, mp3, flac, ogg"}, 400

    if len(file.read()) > max_file_size:
        return {"error": "File size is over limit. Max size is 5MB."}, 400

    file.seek(0)  # Reset file pointer
    mime = magic.Magic(mime=True)
    file_format = mime.from_buffer(file.read(1024))
    if 'audio' not in file_format:
        return {"error": "Invalid file content."}, 400

    file.seek(0)  # Reset file pointer again
    return None

# Routes
@app.route('/upload_audio/', methods=['POST'])
def upload_audio():
    audio_file = request.files.get('file')
    audio_file_label = request.form.get('audio_file_label')

    if not audio_file or not audio_file_label:
        return {"error": "Missing file or audio_file_label."}, 400

    # Validate file
    error = validate_audio_file(audio_file)
    if error:
        return error

    os.makedirs('resources', exist_ok=True)
    file_extension = secure_filename(audio_file.filename).split('.')[-1]
    stored_file_name = f"{audio_file_label}.{file_extension}"

    save_path = os.path.join('resources', stored_file_name)
    audio_file.save(save_path)
    return {"message": f"File {audio_file.filename} uploaded successfully with label {audio_file_label}."}, 200

@app.route('/base_tts/', methods=['GET'])
def base_tts():
    text = request.args.get('text', '')
    accent = request.args.get('accent', 'en-newest')
    speed = float(request.args.get('speed', 1.0))

    if accent not in model:
        return {"error": f"Accent {accent} not found."}, 400

    try:
        save_path = f'{output_dir}/output_v2_{accent}.wav'
        model[accent].tts_to_file(text, model[accent].hps.data.spk2id[key_map[accent][0]], save_path, speed=speed)
        return send_file(save_path, as_attachment=True, mimetype="audio/wav")
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/change_voice/', methods=['POST'])
def change_voice():
    reference_speaker = request.form.get('reference_speaker')
    watermark = request.form.get('watermark', '@MyShell')
    file = request.files.get('file')

    if not reference_speaker or not file:
        return {"error": "Missing reference_speaker or file."}, 400

    error = validate_audio_file(file)
    if error:
        return error

    try:
        file.seek(0)
        temp_file = io.BytesIO(file.read())

        matching_files = [file for file in os.listdir("resources") if file.startswith(reference_speaker)]
        if not matching_files:
            return {"error": "No matching reference speaker found."}, 400

        reference_speaker_file = os.path.join('resources', matching_files[0])
        target_se, audio_name = se_extractor.get_se(reference_speaker_file, tone_color_converter, target_dir='processed', vad=True)

        save_path = f'{output_dir}/output_v2_{reference_speaker}.wav'
        tone_color_converter.convert(
            audio_src_path=temp_file,
            src_se=source_se['en-newest'],
            tgt_se=target_se,
            output_path=save_path,
            message=watermark
        )
        return send_file(save_path, as_attachment=True, mimetype="audio/wav")
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/synthesize_speech/', methods=['GET'])
def synthesize_speech():
    text = request.args.get('text', '')
    voice = request.args.get('voice', '')
    accent = request.args.get('accent', 'en-newest')
    speed = float(request.args.get('speed', 1.0))
    watermark = request.args.get('watermark', '@MyShell')

    if accent not in model:
        return {"error": f"Accent {accent} not found."}, 400

    try:
        matching_files = [file for file in os.listdir("resources") if file.startswith(voice)]
        if not matching_files:
            return {"error": "No matching voice found."}, 400

        reference_speaker = os.path.join('resources', matching_files[0])
        target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, target_dir='processed', vad=True)

        src_path = f'{output_dir}/tmp.wav'
        save_path = f'{output_dir}/output_v2_{accent}.wav'
        model[accent].tts_to_file(text, model[accent].hps.data.spk2id[key_map[accent][0]], src_path, speed=speed)

        tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=source_se[accent],
            tgt_se=target_se,
            output_path=save_path,
            message=watermark
        )

        return send_file(save_path, as_attachment=True, mimetype="audio/wav")
    except Exception as e:
        return {"error": str(e)}, 500

# Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)