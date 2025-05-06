from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

class LieDetectionVideoDataset(Dataset):
    def __init__(self, annotations_file, img_dir, num_frames=16, frame_size=224):
        self.annos = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize((frame_size, frame_size)),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        clip_name = self.annos.iloc[idx, 0]
        label = 1 if self.annos.iloc[idx, 1].lower() in ['lie', 'deception', '1'] else 0

        frame_path = os.path.join(self.img_dir, clip_name)
        frame_names = sorted([f for f in os.listdir(frame_path) if f.endswith(".jpg")])

        # Asegurarse de que tenemos suficientes frames
        if len(frame_names) < self.num_frames:
            # Duplicar el último frame si no hay suficientes
            frame_names = frame_names + [frame_names[-1]] * (self.num_frames - len(frame_names))
        
        indices = np.linspace(0, len(frame_names) - 1, self.num_frames).astype(int)
        frames = []
        for i in indices:
            img_path = os.path.join(frame_path, frame_names[i])
            image = Image.open(img_path).convert("RGB")
            frames.append(self.transforms(image))
        
        frames = torch.stack(frames)  # (T, C, H, W)

        return frames, label
