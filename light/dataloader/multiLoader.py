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
    def __init__(
        self,
        annotations_file,
        wav_dir,
        img_dir,
        num_frames=32,
        frame_size=224,
        max_duration=10.0,
        sample_rate=16000
    ):
        # Read CSV with no header and assign columns
        self.annos = pd.read_csv(
            annotations_file,
            header=None,
            names=["video_id", "clip_name", "start", "end", "label"]
        )
        self.wav_dir = wav_dir
        self.img_dir = img_dir
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        # Audio feature extractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
        # Video frame transforms
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((frame_size, frame_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        row = self.annos.iloc[idx]
        clip_name = row['clip_name'].strip()
        raw_label = row['label'].strip().lower()
        # 1 for lie, 0 for truth
        label = 1 if raw_label == 'lie' else 0

        # ----- Video -----
        frame_path = os.path.join(self.img_dir, clip_name)
        frame_names = sorted([f for f in os.listdir(frame_path) if f.endswith('.jpg')])
        # pad if too few frames
        if len(frame_names) < self.num_frames:
            frame_names += [frame_names[-1]] * (self.num_frames - len(frame_names))
        # uniform sampling
        indices = np.linspace(0, len(frame_names) - 1, self.num_frames).astype(int)
        frames = []
        for i in indices:
            img = Image.open(os.path.join(frame_path, frame_names[i])).convert('RGB')
            frames.append(self.transforms(img))
        video_data = torch.stack(frames)  # [T, 3, H, W]

        # ----- Audio -----
        audio_path = os.path.join(self.wav_dir, clip_name + '.wav')
        waveform, sr = torchaudio.load(audio_path)
        # resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        # mono mix
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # feature extraction
        audio_inputs = self.feature_extractor(
            waveform.squeeze(0).numpy(),
            sampling_rate=self.sample_rate,
            return_tensors='pt',
            padding='max_length',
            max_length=int(self.sample_rate * self.max_duration),
            truncation=True
        )
        audio_data = audio_inputs['input_values'].squeeze(0)  # [max_len]

        return video_data, audio_data, label