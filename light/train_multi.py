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

model_name = "MultimodalLieDetector"  # Nombre para tu modelo
output_dir = "model_checkpoints"  # Directorio base para guardar los modelos
os.makedirs(output_dir, exist_ok=True)  # Crear directorio si no existe


train, test = ['train_fold3.csv', 'test_fold3.csv']
img_dir = "C:/Users/victo/deceptionDetector/dataset/face_frames/"
wav_dir = "C:/Users/victo/deceptionDetector/dataset/audio_files/"
data_root = "C:/Users/victo/deceptionDetector/dataset/DOLOS/"

train_anno = data_root + 'Training_Protocols/' + train
test_anno = data_root + 'Training_Protocols/' + test

train_dataset = LieDetectionMultimodalDataset(train_anno, wav_dir, img_dir, num_frames=24, frame_size=224)
test_dataset = LieDetectionMultimodalDataset(test_anno, wav_dir, img_dir, num_frames=24, frame_size=224)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = VideoMAEConfig.from_pretrained("MCG-NJU/videomae-base", output_hidden_states=True)

config.num_channels = 3
config.image_size = 224 
config.num_frames = 24


video_model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", config=config)
audio_model = WavLMLieDetector()

model = MultimodalLieDetector(video_model, audio_model)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for idx, batch in enumerate(train_loader):
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

    # Evaluación
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            video_inputs, audio_inputs, labels = batch
            video_inputs = video_inputs.to(device)
            audio_inputs = audio_inputs.to(device)
            labels = labels.to(device)
            outputs = model(video_inputs, audio_inputs)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Métricas adicionales
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    avg_test_loss = test_loss / len(test_loader)
    accuracy = correct / total
    
    # Crear directorio para este epoch
    epoch_dir = os.path.join(output_dir, f"{model_name}_epoch{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    # Guardar el modelo
    model_path = os.path.join(epoch_dir, f"{model_name}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, model_path)
    print(f"Model saved to {model_path}")
    
    # Guardar resultados de evaluación
    results_path = os.path.join(epoch_dir, "evaluation_results.txt")
    with open(results_path, 'w') as f:
        f.write(f"Evaluation Results - Epoch {epoch + 1}\n")
        f.write("="*40 + "\n\n")
        f.write(f"Train Loss: {avg_loss:.4f}\n")
        f.write(f"Test Loss: {avg_test_loss:.4f}\n")
        f.write(f"Accuracy: {accuracy:.2%}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        
        # Reporte de clasificación completo
        f.write("Classification Report:\n")
        f.write(classification_report(all_labels, all_preds, target_names=['Truth', 'Lie']))
    
    print(f"Test Loss: {avg_test_loss:.4f} | Accuracy: {accuracy:.2%} | F1: {f1:.4f}\n")
    print(f"Evaluation results saved to {results_path}\n")