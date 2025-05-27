import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import os
import csv
from sklearn.metrics import precision_recall_fscore_support, classification_report
from transformers import VideoMAEForVideoClassification, VideoMAEConfig
from dataloader.multiLoader import LieDetectionMultimodalDataset
from models.lieModelv2 import MultimodalLieDetectorv2
from models.wavLM import WavLMLieDetector

# --- Configuración inicial ---
model_name = "MultimodalLieDetectorv2"
output_dir = "model_checkpoints"
os.makedirs(output_dir, exist_ok=True)

img_dir    = "C:/Users/victo/deceptionDetector/dataset/face_frames/"
wav_dir    = "C:/Users/victo/deceptionDetector/dataset/audio_files/"
data_root  = "C:/Users/victo/deceptionDetector/dataset/DOLOS/"

train_anno = "C:/Users/victo/deceptionDetector/dataset/train.csv"
test_anno  = "C:/Users/victo/deceptionDetector/dataset/test.csv"

train_dataset = LieDetectionMultimodalDataset(train_anno, wav_dir, img_dir, num_frames=24, frame_size=224)
test_dataset  = LieDetectionMultimodalDataset(test_anno,  wav_dir, img_dir, num_frames=24, frame_size=224)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Modelos ---
config = VideoMAEConfig.from_pretrained("MCG-NJU/videomae-base", output_hidden_states=True)
config.num_channels = 3
config.image_size   = 224 
config.num_frames   = 24

video_model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", config=config)
audio_model = WavLMLieDetector()
model       = MultimodalLieDetectorv2(video_model, audio_model, dropout_prob=0.5).to(device)

# --- Optimizador y scheduler ---
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)
loss_fn   = nn.CrossEntropyLoss()

# --- Checkpoint inicial ---
checkpoint_path = os.path.join(output_dir, f"{model_name}_best.pth")
start_epoch     = 0
best_test_loss  = float('inf')

if os.path.isfile(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    best_test_loss = ckpt.get('test_loss', best_test_loss)
    start_epoch    = ckpt.get('epoch', 0)
    print(f"=> Loaded checkpoint (epoch {start_epoch}, test_loss {best_test_loss:.4f})")
else:
    print("=> No checkpoint found, training from scratch.")

# --- Preparar archivo de métricas ---
metrics_file = os.path.join(output_dir, f"{model_name}_metrics.csv")
with open(metrics_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'test_loss', 'accuracy', 'precision', 'recall', 'f1'])

# --- Loop de entrenamiento ---
patience   = 5
no_improve = 0
num_epochs = 20

for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0

    for idx, (video_inputs, audio_inputs, labels) in enumerate(train_loader):
        video_inputs = video_inputs.to(device)
        audio_inputs = audio_inputs.to(device)
        labels       = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(video_inputs, audio_inputs)
        loss    = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"Epoch {epoch+1}, Batch {idx+1}/{len(train_loader)} — Loss: {loss.item():.4f}", end='\r')

    avg_train_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch+1} Completed | Avg Train Loss: {avg_train_loss:.4f}")

    # --- Validación ---
    model.eval()
    test_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for video_inputs, audio_inputs, labels in test_loader:
            video_inputs = video_inputs.to(device)
            audio_inputs = audio_inputs.to(device)
            labels       = labels.to(device, dtype=torch.long)

            outputs = model(video_inputs, audio_inputs)
            loss    = loss_fn(outputs, labels)
            test_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_test_loss = test_loss / len(test_loader)
    scheduler.step(avg_test_loss)

    # --- Guardar modelo cada epoch ---
    epoch_checkpoint = os.path.join(output_dir, f"{model_name}_epoch{epoch+1}.pth")
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_loss': avg_test_loss,
    }, epoch_checkpoint)
    print(f"=> Saved epoch checkpoint: {epoch+1} (test_loss {avg_test_loss:.4f})")

    # --- Early stopping y best model ---
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        no_improve     = 0
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_loss': avg_test_loss,
        }, checkpoint_path)
        print(f"=> New best model saved at epoch {epoch+1} (test_loss {avg_test_loss:.4f})")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"=> Early stopping after {patience} epochs without improvement.")
            break

    # --- Cálculo de métricas y reporte ---
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    accuracy = correct / total
    print(f"Test Loss: {avg_test_loss:.4f} | Acc: {accuracy:.2%} | F1: {f1:.4f}")
    print(classification_report(all_labels, all_preds, target_names=['Truth','Lie']))

    # --- Anotar métricas en CSV ---
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, f"{avg_train_loss:.4f}", f"{avg_test_loss:.4f}",
                         f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"])

print("Training finished.")
