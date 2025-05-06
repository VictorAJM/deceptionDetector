import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from dataloader.audioLoader import LieDetectionAudioDataset
from models.wavLM import WavLMLieDetector

import os
import csv

def build_labels_dict(csv_path):
    labels_dict = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            filename = row[0].strip() + ".wav"
            label = 1 if row[1].strip().lower() == "deception" else 0
            labels_dict[filename] = label
    return labels_dict

train, test = ['train_fold3.csv', 'test_fold3.csv']
wav_dir = "C:/Users/victo/deceptionDetector/dataset/audio_files/"
data_root = "C:/Users/victo/deceptionDetector/dataset/DOLOS/"

train_anno = data_root + 'Training_Protocols/' + train
test_anno = data_root + 'Training_Protocols/' + test

train_labels_dict = build_labels_dict(train_anno)
test_labels_dict = build_labels_dict(test_anno)

train_dataset = LieDetectionAudioDataset(wav_dir, train_labels_dict)
test_dataset = LieDetectionAudioDataset(wav_dir, test_labels_dict)

# Cargar datos
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# Modelo
model = WavLMLieDetector()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizador y configuración
optimizer = optim.Adam(model.parameters(), lr=2e-5)
num_epochs = 5

# Entrenamiento
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for idx, batch in enumerate(train_loader):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
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
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            test_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_test_loss = test_loss / len(test_loader)
    accuracy = correct / total
    print(f"Test Loss: {avg_test_loss:.4f} | Accuracy: {accuracy:.2%}\n")

