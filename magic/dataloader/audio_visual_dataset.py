import os
import pandas as pd
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchaudio
import torchaudio.functional as F

class AudioVisualDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, img_dir, num_tokens=64, frame_size=224):
        super(AudioVisualDataset, self).__init__()

        self.annos = pd.read_csv(
            annotations_file,
            header=None,
            names=["video_id", "clip_name", "start", "end", "label"]
        )
        self.audio_dir = audio_dir  # all files in '.wav' format
        self.num_tokens = num_tokens

        self.img_dir = img_dir
        self.frame_size = frame_size  # 224 or any multiple of 32
        self.transforms = T.Compose([
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Resize(frame_size),
            # normalize to imagenet mean and std values
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.number_of_target_frames = num_tokens

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):

        # select one clip
        row = self.annos.iloc[idx]
        clip_name = row['clip_name'].strip()
        # get audio path
        audio_path = os.path.join(self.audio_dir, clip_name + '.wav')

        # load the audio file with torch audio
        waveform, sample_rate = torchaudio.load(audio_path)
        # use mono audio instead os stereo audio (use left by default)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # calculate duration of the audio clip
        clip_duration = len(waveform) / sample_rate
        """
        # for wav2vec2, 1 Token corresponds to ~ 321.89 discrete samples
        # to get precisely 64 tokens (a hyperparameter that can be changed), the length of input discrete samples to the model should be 321.89 * 64
        # divide the above by the clip duration to get new sample rate (or) new_sample_rate * clip_duration = 321.89 * num tokens
        """
        clip_duration = len(waveform[0]) / sample_rate
        new_sample_rate = int(321.89 * 64 / clip_duration)
        # required by collate function
        waveform = F.resample(waveform, sample_rate, new_sample_rate)

        #  get face feature path
        file_path = self.img_dir + clip_name + '/'
        # list all jpeg images
        frame_names = [i for i in os.listdir(file_path) if i.split('.')[-1] == 'jpg']

        # sample 64 face frames
        target_frames = np.linspace(0, len(frame_names) - 1, num=self.number_of_target_frames)
        target_frames = np.around(target_frames).astype(
            int)  # certain frames may be redundant because of rounding to the nearest integer
        face_frames = []
        for i in target_frames:
            img = np.asarray(Image.open(file_path + frame_names[i])) / 255.0
            face_frames.append(self.transforms(img))
        face_frames = torch.stack(face_frames, 0)
        face_frames.type(torch.float32)

        # assign integer to labels
        raw_label = row['label'].strip().lower()
        # 1 for lie, 0 for truth
        label = 1 if raw_label == 'lie' else 0

        return waveform, face_frames, label


def af_pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def af_collate_fn(batch):
    tensors, face_tensors, targets = [], [], []

    # Gather in lists, and encode labels as indices
    for waveform, face_frames, label in batch:
        tensors += [waveform]
        face_tensors += [face_frames]
        targets += [torch.tensor(label)]

    # Group the list of tensors into a batched tensor
    tensors = af_pad_sequence(tensors)
    face_tensors = torch.stack(face_tensors)
    targets = torch.stack(targets)

    return tensors, face_tensors, targets
