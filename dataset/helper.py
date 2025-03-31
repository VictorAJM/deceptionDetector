import torch
import torchaudio.functional as F
audio_path = "C:/Users/victo/deceptionDetector/dataset/audio_files/AN_WILTY_EP28_truth7.wav"
import torchaudio
waveform, sample_rate = torchaudio.load(audio_path)

# Convertir a mono si es estéreo
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

# Calcular nueva tasa de muestreo
clip_duration = len(waveform[0]) / sample_rate
new_sample_rate = int(321.89 * 64 / clip_duration)

# Resamplear audio
waveform = F.resample(waveform, sample_rate, new_sample_rate)
print(f"✅ Audio resampleado a {new_sample_rate} Hz. Nueva forma: {waveform.shape}")