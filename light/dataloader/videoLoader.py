from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

class LieDetectionVideoDataset(Dataset):
    def __init__(self, annotations_file, img_dir, num_frames=16, frame_size=224):
        # Read with no header and assign meaningful names
        self.annos = pd.read_csv(
            annotations_file,
            header=None,
            names=['video_id', 'clip_name', 'start', 'end', 'label']
        )
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
        # use the clip_name (column 1) for your folder of extracted frames
        clip_name = self.annos.iloc[idx]['clip_name'].strip()
        # label is in column 'label' (column 4) → 1 for 'lie', 0 for 'truth'
        raw_label = self.annos.iloc[idx]['label'].strip().lower()
        label = 1 if raw_label == 'lie' else 0

        frame_path = os.path.join(self.img_dir, clip_name)
        frame_names = sorted(f for f in os.listdir(frame_path) if f.endswith(".jpg"))

        # pad by repeating last frame if too few
        if len(frame_names) < self.num_frames:
            frame_names += [frame_names[-1]] * (self.num_frames - len(frame_names))

        # uniformly sample num_frames
        indices = np.linspace(0, len(frame_names) - 1, self.num_frames).astype(int)
        frames = []
        for i in indices:
            img = Image.open(os.path.join(frame_path, frame_names[i])).convert("RGB")
            frames.append(self.transforms(img))

        frames = torch.stack(frames)  # shape: [num_frames, 3, H, W]
        return frames, label
