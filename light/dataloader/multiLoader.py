import os
import torch
from torch.utils.data import Dataset
import torchaudio
from PIL import Image
import pandas as pd
import numpy as np
from torchvision import transforms
from transformers import AutoFeatureExtractor

class LieDetectionMultimodalDataset(Dataset):
    def __init__(self, annotations_file, wav_dir, img_dir, num_frames=32, frame_size=224, max_duration=10.0, sample_rate=16000):
        self.annos = pd.read_csv(annotations_file)
        self.wav_dir = wav_dir
        self.img_dir = img_dir
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((frame_size, frame_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        clip_name = self.annos.iloc[idx, 0]
        label = 1 if self.annos.iloc[idx, 1].lower() in ['lie', 'deception', '1'] else 0

        # Load video frames
        frame_path = os.path.join(self.img_dir, clip_name)
        frame_names = sorted([f for f in os.listdir(frame_path) if f.endswith(".jpg")])

        if len(frame_names) < self.num_frames:
            frame_names = frame_names + [frame_names[-1]] * (self.num_frames - len(frame_names))
        
        indices = np.linspace(0, len(frame_names) - 1, self.num_frames).astype(int)
        frames = []
        for i in indices:
            img_path = os.path.join(frame_path, frame_names[i])
            image = Image.open(img_path).convert("RGB")
            frames.append(self.transforms(image))
        
        video_data = torch.stack(frames)

        # Load audio
        audio_path = os.path.join(self.wav_dir, clip_name + ".wav")
        waveform, sr = torchaudio.load(audio_path)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        audio_inputs = self.feature_extractor(
            waveform.squeeze(0).numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=int(self.sample_rate * self.max_duration),
            truncation=True
        )

        return video_data, audio_inputs["input_values"].squeeze(0), label