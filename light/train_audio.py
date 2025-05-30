import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from dataloader.audioLoader import LieDetectionAudioDataset
from models.wavLM import WavLMLieDetector
from dataloader.multiLoader import LieDetectionMultimodalDataset

import os
import csv

def build_labels_dict(csv_path):
    labels_dict = {}
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            # row[1] is your unique segment name, e.g. “YW_WILTY_EP49_lie4”
            filename = row[1].strip() + ".wav"
            # row[4] is your label (“lie” or “truth”)
            label = 1 if row[4].strip().lower() == "lie" else 0
            labels_dict[filename] = label
    return labels_dict

model_name = "AudioLieDetector"
output_dir = "model_checkpoints"
os.makedirs(output_dir, exist_ok=True)

img_dir = "C:/Users/victo/deceptionDetector/dataset/face_frames/"
wav_dir = "C:/Users/victo/deceptionDetector/dataset/audio_files/"
data_root = "C:/Users/victo/deceptionDetector/dataset/DOLOS/"

train_anno = "C:/Users/victo/deceptionDetector/dataset/train.csv"
test_anno = "C:/Users/victo/deceptionDetector/dataset/test.csv"

train_dataset = LieDetectionMultimodalDataset(train_anno, wav_dir, img_dir, num_frames=1, frame_size=224)
test_dataset = LieDetectionMultimodalDataset(test_anno, wav_dir, img_dir, num_frames=1, frame_size=224)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


# Modelo
model = WavLMLieDetector()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- Optimizador con Weight Decay (L2) y LR Scheduler ---
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)  # Añadido L2
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

loss_fn = nn.CrossEntropyLoss()

# --- Early Stopping ---
best_test_loss = float('inf')
patience = 5  # Número de epochs sin mejora antes de parar
no_improve = 0
num_epochs = 20

# Entrenamiento
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for idx, batch in enumerate(train_loader):
        video_inputs, audio_inputs, labels = batch
        audio_inputs = audio_inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(audio_inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Batch {idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}", end='\r')

    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch + 1} Completed | Average Train Loss: {avg_loss:.4f}")

    # --- Evaluación ---
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            video_inputs, audio_inputs, labels = batch
            audio_inputs = audio_inputs.to(device)
            labels = labels.to(device)
            outputs = model(audio_inputs)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    scheduler.step(avg_test_loss)  # Ajusta el LR si no hay mejora

    # --- Early Stopping Check ---
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        no_improve = 0
        # Guardar el mejor modelo
        best_model_path = os.path.join(output_dir, f"{model_name}_best.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'test_loss': avg_test_loss,
        }, best_model_path)
        print(f"Best model saved to {best_model_path}")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break

    # --- Guardar métricas ---
    epoch_dir = os.path.join(output_dir, f"{model_name}_epoch{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    model_path = os.path.join(epoch_dir, f"{model_name}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, model_path)
    print(f"Model saved to {model_path}")

    # --- Resultados ---
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    accuracy = correct / total
    
    results_path = os.path.join(epoch_dir, "evaluation_results.txt")
    with open(results_path, 'w') as f:
        f.write(f"Epoch {epoch + 1}\n")
        f.write(f"Train Loss: {avg_loss:.4f}\n")
        f.write(f"Test Loss: {avg_test_loss:.4f}\n")
        f.write(f"Accuracy: {accuracy:.2%}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(classification_report(all_labels, all_preds, target_names=['Truth', 'Lie']))
    
    print(f"Test Loss: {avg_test_loss:.4f} | Accuracy: {accuracy:.2%} | F1: {f1:.4f}\n")