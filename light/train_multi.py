import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from dataloader.audioLoader import LieDetectionAudioDataset
from models.lieModel import MultimodalLieDetector
from models.wavLM import WavLMLieDetector

import os
import csv

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


from dataloader.videoLoader import LieDetectionVideoDataset

from transformers import VideoMAEForVideoClassification, VideoMAEFeatureExtractor, VideoMAEConfig
import torch.nn as nn


from dataloader.multiLoader import LieDetectionMultimodalDataset


train, test = ['train_fold3.csv', 'test_fold3.csv']
img_dir = "C:/Users/victo/deceptionDetector/dataset/face_frames/"
wav_dir = "C:/Users/victo/deceptionDetector/dataset/audio_files/"
data_root = "C:/Users/victo/deceptionDetector/dataset/DOLOS/"

train_anno = data_root + 'Training_Protocols/' + train
test_anno = data_root + 'Training_Protocols/' + test

train_dataset = LieDetectionMultimodalDataset(train_anno, wav_dir, img_dir, num_frames=32, frame_size=224)
test_dataset = LieDetectionMultimodalDataset(test_anno, wav_dir, img_dir, num_frames=32, frame_size=224)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = VideoMAEConfig.from_pretrained("MCG-NJU/videomae-base", output_hidden_states=True)

config.num_channels = 3
config.image_size = 224 
config.num_frames = 32


video_model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", config=config)
audio_model = WavLMLieDetector()

model = MultimodalLieDetector(video_model, audio_model)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

num_epochs = 15

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for idx, batch in enumerate(train_loader):
        if idx > 5: continue
        video_inputs, audio_inputs, labels = batch
        video_inputs = video_inputs.to(device)
        audio_inputs = audio_inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(video_inputs, audio_inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Batch {idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}", end='\r')

    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch + 1} Completed | Average Train Loss: {avg_loss:.4f}")

    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for video_inputs, audio_inputs, labels in test_loader:
            video_inputs = video_inputs.to(device)
            audio_inputs = audio_inputs.to(device)
            labels = labels.to(device)
            outputs = model(video_inputs, audio_inputs)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_test_loss = test_loss / len(test_loader)
    accuracy = correct / total
    print(f"Test Loss: {avg_test_loss:.4f} | Accuracy: {accuracy:.2%}\n")