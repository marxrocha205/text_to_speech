import os
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# Definir os caminhos
output_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = "C:/Users/Marx/Downloads/text-generation-webui-main/text-generation-webui-main/extensions/coqui_tts/data/"

# Configurar o dataset
dataset_config = BaseDatasetConfig(
    formatter="custom",  # Ajuste isso conforme necessário
    meta_file_train="metadata.csv",  # Certifique-se de que o arquivo metadata.csv está no caminho correto
    path=dataset_path
)

# Configurar o áudio
audio_config = BaseAudioConfig(
    sample_rate=22050,
    resample=True,
    do_trim_silence=True,
    trim_db=23.0
)

# Configurar o modelo
config = GlowTTSConfig(
    batch_size=64,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    precompute_num_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="phoneme_cleaners",
    use_phonemes=True,
    phoneme_language="pt-br",  # Ajuste conforme o idioma
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    use_speaker_embedding=True,  # Ajuste conforme necessário
    min_text_len=0,
    max_text_len=500,
    min_audio_len=0,
    max_audio_len=500000,
)

# Inicializar o processador de áudio
ap = AudioProcessor.init_from_config(config)

# Inicializar o tokenizador
tokenizer, config = TTSTokenizer.init_from_config(config)

# Carregar amostras de dados
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# Inicializar o gerenciador de locutores para treinamento multi-locutor
speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.num_speakers = speaker_manager.num_speakers

# Inicializar o modelo
model = GlowTTS(config, ap, tokenizer, speaker_manager=speaker_manager)

# Inicializar o treinador
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)

# Iniciar o treinamento
trainer.fit()
