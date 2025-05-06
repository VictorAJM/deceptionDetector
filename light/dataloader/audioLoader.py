import os
import torch
from torch.utils.data import Dataset
import torchaudio
from transformers import AutoFeatureExtractor

class LieDetectionAudioDataset(Dataset):
    def __init__(self, wav_dir, labels_dict, max_duration=10.0, sample_rate=16000):
        self.wav_dir = wav_dir
        self.labels_dict = labels_dict
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
        self.filenames = list(labels_dict.keys())

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        label = self.labels_dict[filename]

        # Load audio
        path = os.path.join(self.wav_dir, filename)
        waveform, sr = torchaudio.load(path)

        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # Convert to mono if multi-channel
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Extract features
        inputs = self.feature_extractor(
            waveform.squeeze(0).numpy(),  # Convert to numpy array
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=int(self.sample_rate * self.max_duration),
            truncation=True
        )

        return inputs["input_values"].squeeze(0), torch.tensor(label, dtype=torch.long)