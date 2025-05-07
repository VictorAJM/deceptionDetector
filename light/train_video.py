import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


from dataloader.videoLoader import LieDetectionVideoDataset

from transformers import VideoMAEForVideoClassification, VideoMAEFeatureExtractor, VideoMAEConfig
import torch.nn as nn



train, test = ['train_fold3.csv', 'test_fold3.csv']
img_dir = "C:/Users/victo/deceptionDetector/dataset/face_frames/"
data_root = "C:/Users/victo/deceptionDetector/dataset/DOLOS/"

train_anno = data_root + 'Training_Protocols/' + train
test_anno = data_root + 'Training_Protocols/' + test

train_dataset = LieDetectionVideoDataset(train_anno, img_dir, num_frames=32)
test_dataset = LieDetectionVideoDataset(test_anno, img_dir, num_frames=32)

# Cargar datos
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Batch size reducido
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
# Model & setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = VideoMAEConfig.from_pretrained("MCG-NJU/videomae-base")
config.num_channels = 3
config.image_size = 224 
config.num_frames = 32

model = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-base",
    config=config,
    ignore_mismatched_sizes=True
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()
# Training loop
for epoch in range(3):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for idx, batch in enumerate(train_loader):
        videos, labels = batch
        videos = videos.to(device)
        labels = labels.to(device)

        outputs = model(videos)
        loss = loss_fn(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        print(f"Epoch {epoch + 1}, Batch {idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}", end='\r')

    train_acc = accuracy_score(all_labels, all_preds)
    print(f"\nEpoch {epoch+1}, Train Loss: {total_loss:.4f}, Train Accuracy: {train_acc:.4f}")

    # Evaluación
    model.eval()
    test_preds, test_labels = [], []

    with torch.no_grad():
        for videos, labels in test_loader:
            labels = labels.to(device)
            videos = videos.to(device)
            outputs = model(videos)
            preds = outputs.logits.argmax(dim=1)

            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='binary')
    print(f"Eval Accuracy: {test_acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}\n")