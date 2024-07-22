import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path
import threading
import subprocess
import os
import torch
from TTS.api import TTS

# Configurações do modelo TTS
params = {
    "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

this_dir = str(Path(__file__).parent.resolve())
voice_dir = Path(f"{this_dir}/voices")
output_dir = Path(f"{this_dir}/outputs")
output_dir.mkdir(parents=True, exist_ok=True)  # Cria a pasta outputs se não existir
audio_counter = {lang: 1 for lang in ["pt", "en", "es", "fr", "de", "ru"]}
selected_voice = "default.wav"

def load_model():
    model = TTS(params["model_name"]).to(params["device"])
    return model

def apply_emotion_settings(emotion):
    if emotion == "Triste":
        pitch_scale.set(-20)
        speed_scale.set(-20)
        volume_scale.set(-10)
    elif emotion == "Raiva":
        pitch_scale.set(20)
        speed_scale.set(20)
        volume_scale.set(10)
    elif emotion == "Feliz":
        pitch_scale.set(15)
        speed_scale.set(15)
        volume_scale.set(10)
    elif emotion == "Surpreso":
        pitch_scale.set(10)
        speed_scale.set(10)
        volume_scale.set(10)
    elif emotion == "Cansado":
        pitch_scale.set(-10)
        speed_scale.set(-10)
        volume_scale.set(-10)
    elif emotion == "Neutro":
        pitch_scale.set(0)
        speed_scale.set(0)
        volume_scale.set(0)
    else:
        pitch_scale.set(0)
        speed_scale.set(0)
        volume_scale.set(0)

def convert_text_to_speech():
    text = text_entry.get("1.0", tk.END).strip()
    pitch = pitch_scale.get()
    speed = speed_scale.get()
    volume = volume_scale.get()
    language = language_var.get()
    emotion = emotion_var.get()
    split_sentences = split_sentences_var.get()

    if not text:
        messagebox.showwarning("Entrada Inválida", "O campo de texto não pode estar vazio.")
        return

    def conversion_task():
        show_loading_screen()

        model = load_model()

        global audio_counter
        file_suffix = f"audio_{audio_counter[language]}_{language}.wav"
        output_file = output_dir / file_suffix
        
        speaker_wav = Path(f'{voice_dir}/{selected_voice}')

        ssml_text = f'<prosody pitch="{pitch}%" rate="{speed}%" volume="{volume}">{text}</prosody>'

        model.tts_to_file(
            text=ssml_text,
            speaker_wav=str(speaker_wav),
            language=language,
            emotion=emotion,
            speed=speed,
            file_path=str(output_file),
            split_sentences=split_sentences
        )

        audio_counter[language] += 1
        hide_loading_screen()
        messagebox.showinfo("Conversão Concluída", f"Texto convertido para voz. Arquivo salvo em: {output_file}")
        open_audio_file(output_file)

    threading.Thread(target=conversion_task).start()

def show_loading_screen():
    global loading_screen
    loading_screen = tk.Toplevel(root)
    loading_screen.title("Carregando")
    loading_label = ttk.Label(loading_screen, text="Carregando, por favor aguarde...")
    loading_label.pack(padx=20, pady=20)
    loading_screen.grab_set()

def hide_loading_screen():
    loading_screen.grab_release()
    loading_screen.destroy()

def open_audio_file(file_path):
    if os.name == 'nt':  # Para Windows
        os.startfile(file_path)
    else:
        subprocess.call(('xdg-open', file_path))

def load_voices():
    voices = [f.name for f in voice_dir.glob("*.wav")]
    voice_combobox['values'] = voices
    if voices:
        voice_combobox.set(voices[0])

def on_voice_select(event):
    global selected_voice
    selected_voice = voice_combobox.get()

# Configuração da janela principal
root = tk.Tk()
root.title("Conversor de Texto para Fala")

# Layout principal
main_frame = ttk.Frame(root, padding="10")
main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Expandir e preencher o espaço disponível
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
main_frame.columnconfigure(1, weight=1)

# Campo de entrada de texto
tk.Label(main_frame, text="Digite o texto:").grid(row=0, column=0, columnspan=2, pady=5, sticky=tk.W)
text_entry = tk.Text(main_frame, height=10, width=50)
text_entry.grid(row=1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

# Configuração de Pitch
tk.Label(main_frame, text="Pitch (%):").grid(row=2, column=0, pady=5, sticky=tk.W)
pitch_scale = tk.Scale(main_frame, from_=-50, to=50, orient=tk.HORIZONTAL)
pitch_scale.set(0)  # Valor padrão
pitch_scale.grid(row=2, column=1, pady=5, sticky=(tk.W, tk.E))

# Configuração de Velocidade
tk.Label(main_frame, text="Velocidade (%):").grid(row=3, column=0, pady=5, sticky=tk.W)
speed_scale = tk.Scale(main_frame, from_=-50, to=50, orient=tk.HORIZONTAL)
speed_scale.set(0)  # Valor padrão
speed_scale.grid(row=3, column=1, pady=5, sticky=(tk.W, tk.E))

# Configuração de Volume
tk.Label(main_frame, text="Volume:").grid(row=4, column=0, pady=5, sticky=tk.W)
volume_scale = tk.Scale(main_frame, from_=-50, to=50, orient=tk.HORIZONTAL)
volume_scale.set(0)  # Valor padrão
volume_scale.grid(row=4, column=1, pady=5, sticky=(tk.W, tk.E))

# Seleção de Emoção
tk.Label(main_frame, text="Selecione a Emoção:").grid(row=5, column=0, pady=5, sticky=tk.W)
emotion_var = tk.StringVar(value="Normal")
emotion_options = ["Normal", "Triste", "Raiva", "Feliz", "Surpreso", "Cansado"]
emotion_menu = ttk.OptionMenu(main_frame, emotion_var, *emotion_options, command=apply_emotion_settings)
emotion_menu.grid(row=5, column=1, pady=5, sticky=(tk.W, tk.E))

# Seleção de Língua
tk.Label(main_frame, text="Selecione a Língua:").grid(row=6, column=0, pady=5, sticky=tk.W)
language_var = tk.StringVar(value="pt")
language_options = ["pt", "en", "es", "fr", "de", "ru", "pt"]
language_menu = ttk.OptionMenu(main_frame, language_var, *language_options)
language_menu.grid(row=6, column=1, pady=5, sticky=(tk.W, tk.E))

# Seleção de Voz
tk.Label(main_frame, text="Selecione a Voz:").grid(row=7, column=0, pady=5, sticky=tk.W)
voice_combobox = ttk.Combobox(main_frame, state="readonly", width=30)
voice_combobox.grid(row=7, column=1, pady=5, sticky=(tk.W, tk.E))
voice_combobox.bind("<<ComboboxSelected>>", on_voice_select)

load_voices()  # Carrega as vozes ao iniciar

# Seleção para dividir sentenças
tk.Label(main_frame, text="Dividir Sentenças:").grid(row=8, column=0, pady=5, sticky=tk.W)
split_sentences_var = tk.BooleanVar(value=True)
split_sentences_checkbutton = ttk.Checkbutton(main_frame, text="Dividir Sentenças", variable=split_sentences_var)
split_sentences_checkbutton.grid(row=8, column=1, pady=5, sticky=(tk.W, tk.E))

# Botão para converter texto em fala
convert_button = ttk.Button(main_frame, text="Converter para Fala", command=convert_text_to_speech)
convert_button.grid(row=9, column=0, columnspan=2, pady=20)

# Inicia a aplicação Tkinter
root.mainloop()
