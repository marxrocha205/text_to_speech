import os
from pydub import AudioSegment
from pydub.utils import make_chunks
import speech_recognition as sr

# Função para cortar áudio em segmentos de 10 segundos
def split_audio(input_path, output_dir, chunk_length_ms=10000):
    # Cria a pasta de saída se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Carrega o áudio
    audio = AudioSegment.from_wav(input_path)

    # Divide o áudio em segmentos de 10 segundos
    chunks = make_chunks(audio, chunk_length_ms)

    # Salva cada segmento na pasta de saída
    for i, chunk in enumerate(chunks):
        chunk_name = f"{output_dir}/chunk_{i:04d}.wav"
        chunk.export(chunk_name, format="wav")
        print(f"Exportado {chunk_name}")

# Função para transcrever áudio
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language='pt-BR')
            return text
        except sr.UnknownValueError:
            return "Transcrição não reconhecida"
        except sr.RequestError as e:
            return f"Erro na solicitação: {e}"

# Caminho do arquivo de áudio
input_audio_path = "voices\Jardim das emoçoes.wav"
output_directory = "outputs\wavs"

# Corta o áudio e salva os segmentos
split_audio(input_audio_path, output_directory)

# Transcreve os segmentos e salva as transcrições em um arquivo de texto
with open(f"{output_directory}/transcriptions.txt", "w", encoding="utf-8") as transcriptions_file:
    for chunk_file in sorted(os.listdir(output_directory)):
        if chunk_file.endswith(".wav"):
            chunk_path = os.path.join(output_directory, chunk_file)
            transcription = transcribe_audio(chunk_path)
            transcriptions_file.write(f"{chunk_file}|{transcription}|{transcription}\n")
            print(f"Transcrito {chunk_file}: {transcription}")
