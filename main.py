import json
import os
import html
import torch
from pathlib import Path
from TTS.api import TTS
from TTS.utils.synthesizer import Synthesizer

os.environ["COQUI_TOS_AGREED"] = "1"

params = {
    "voice": "marx.wav",
    "language": "Portuguese",
    "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

this_dir = str(Path(__file__).parent.resolve())
model = None
with open(Path(f"{this_dir}/languages.json"), encoding='utf8') as f:
    languages = json.load(f)


def load_model():
    model = TTS(params["model_name"]).to(params["device"])
    return model


def convert_text_to_voice(text):
    text = html.unescape(text)
    output_file = Path(f'output_audio.wav')

    model.tts_to_file(
        text=text,
        file_path=output_file,
        speaker_wav=[f"{this_dir}/voices/{params['voice']}"],
        language=languages[params["language"]]
    )

    print(f"Texto convertido para voz. Arquivo salvo em: {output_file}")


def setup():
    global model
    print("[XTTS] Carregando o modelo XTTS...")
    model = load_model()
    print("[XTTS] Modelo carregado com sucesso!")


if __name__ == "__main__":
    setup()

    # Texto a ser convertido para voz
    input_text = "O rápido rapaz marrom, saltou sobre o cão preguiçoso!"

    convert_text_to_voice(input_text)
