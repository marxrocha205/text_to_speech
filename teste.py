import soundfile as sf

def load_wav(filename):
    try:
        data, samplerate = sf.read(filename)
        return data, samplerate
        print("achou")
    except Exception as e:
        print(f"Erro ao carregar o arquivo {filename}: {e}")
        return None, None

# Exemplo de uso
audio_data, sample_rate = load_wav('C:/Users/Marx/Downloads/text-generation-webui-main/text-generation-webui-main/extensions/coqui_tts/data/girafali1.wav')
