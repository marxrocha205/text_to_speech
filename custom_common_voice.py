import os
import pandas as pd
from TTS.utils.audio import AudioProcessor
from TTS.tts.datasets import TTSDataset

class CustomCommonVoiceFormatter(TTSDataset):
    def __init__(self, path, meta_file_train, **kwargs):
        self.path = os.path.abspath(path)
        self.meta_file_train = os.path.abspath(meta_file_train)
        self.samples = self.load_samples()
        self.ap = AudioProcessor(
            sample_rate=22050,
            num_mels=80,
            min_level_db=-100,
            frame_shift_ms=12.5,
            frame_length_ms=50.0,
            ref_level_db=20,
            fft_size=1024,
            power=1.5,
            preemphasis=0.0,
            griffin_lim_iters=60,
            signal_norm=True,
            symmetric_norm=True,
            mel_fmin=0,
            mel_fmax=None,
            pitch_fmin=1.0,
            pitch_fmax=640.0,
            spec_gain=20.0,
            stft_pad_mode='reflect',
            max_norm=4.0,
            clip_norm=True,
            do_trim_silence=True,
            trim_db=45,
            do_sound_norm=False,
            do_amp_to_db_linear=True,
            do_amp_to_db_mel=True,
            do_rms_norm=False,
            db_level=None,
            stats_path=None,
            base=10,
            hop_length=256,
            win_length=1024
        )

    def load_samples(self):
        df = pd.read_csv(self.meta_file_train, delimiter=',', header=None, names=["audio_file", "text"])
        samples = []
        for _, row in df.iterrows():
            audio_path = os.path.join(self.path, row['audio_file'])
            text = row['text']
            samples.append({
                "audio_file": audio_path,
                "text": text,
                "speaker_name": None,  # Ajuste se necessário
                "language": "pt-br"    # Ajuste conforme o idioma
            })
        return samples
