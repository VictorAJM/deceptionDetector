import argparse
import os
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_curve, auc

# Importa las mismas clases y funciones que en el script de entrenamiento
from dataloader.audio_visual_dataset import AudioVisualDataset, af_collate_fn
from models.audio_model import W2V2_Model
from models.fusion_model import Fusion
from models.visual_model import ViT_model
from models.improved_model import EnhancedDeceptionDetector

def load_model(args, model_path):
    """Carga el modelo entrenado según los parámetros guardados."""
    if args.model_to_train == 'audio':
        model = W2V2_Model(args.num_encoders, args.adapter, args.adapter_type)
    elif args.model_to_train == 'vision':
        model = ViT_model(args.num_encoders, args.adapter, args.adapter_type)
    else:  # fusion
        model = EnhancedDeceptionDetector()
    
    model.load_state_dict(torch.load(model_path))
    model.to(args.device)
    model.eval()
    return model

def evaluate_model(args, test_loader, model):
    """Evalúa el modelo en el conjunto de prueba."""
    epoch_predictions = []
    epoch_labels = []
    
    with torch.no_grad():
        if args.model_to_train == "audio":
            for waves, _, labels in test_loader:
                waves = waves.squeeze(1).to(args.device)
                labels = labels.to(args.device)
                preds = model(waves)
                epoch_predictions.append(torch.argmax(preds, dim=1))
                epoch_labels.append(labels)
                
        elif args.model_to_train == "vision":
            for _, faces, labels in test_loader:
                faces = faces.to(args.device)
                labels = labels.to(args.device)
                preds = model(faces)
                epoch_predictions.append(torch.argmax(preds, dim=1))
                epoch_labels.append(labels)
                
        else:  # fusion
            for waves, faces, labels in test_loader:
                waves = waves.squeeze(1).to(args.device)
                faces = faces.to(args.device)
                labels = labels.to(args.device)
                preds, _, _ = model(waves, faces)
                epoch_predictions.append(torch.argmax(preds, dim=1))
                epoch_labels.append(labels)
    
    # Concatena todas las predicciones y etiquetas
    epoch_predictions = torch.cat(epoch_predictions).cpu().numpy()
    epoch_labels = torch.cat(epoch_labels).cpu().numpy()
    
    # Calcula métricas
    acc = accuracy_score(epoch_labels, epoch_predictions)
    f1 = f1_score(epoch_labels, epoch_predictions)
    fpr, tpr, _ = roc_curve(epoch_labels, epoch_predictions, pos_label=1)
    auc_score = auc(fpr, tpr)
    report = classification_report(epoch_labels, epoch_predictions, target_names=["truth", "deception"])
    
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc_score:.4f}")
    print("Classification Report:")
    print(report)
    
    return acc, f1, auc_score, report

def main():
    # Configura los argumentos (similar al script de entrenamiento)
    parser = argparse.ArgumentParser(description="Evaluación del modelo entrenado")
    parser.add_argument('--device', type=str, default="cuda:0", help='dispositivo para evaluación')
    parser.add_argument('--batch_size', type=int, default=16, help='tamaño del lote para evaluación')
    parser.add_argument('--data_root', type=str, default='/home/nithish/WILTY/data/', help='directorio raíz de los datos')
    parser.add_argument('--audio_path', type=str, default='/home/nithish/WILTY/data/audio_files/', help='ruta de los archivos de audio')
    parser.add_argument('--visual_path', type=str, default='/home/nithish/WILTY/data/face_frames/', help='ruta de los frames visuales')
    parser.add_argument('--test_file', type=str, required=True, help='archivo CSV con los datos de prueba')
    parser.add_argument('--model_to_train', type=str, required=True, choices=['audio', 'vision', 'fusion'], help='tipo de modelo a evaluar')
    parser.add_argument('--model_path', type=str, required=True, help='ruta al modelo guardado (.pth)')
    parser.add_argument('--num_encoders', type=int, default=4, help='número de codificadores transformer')
    parser.add_argument('--adapter', action='store_true', help='usar adaptadores')
    parser.add_argument('--adapter_type', type=str, default='efficient_conv', help='tipo de adaptador')
    parser.add_argument('--fusion_type', type=str, default='cross2', help='tipo de fusión (solo para modelo fusion)')
    parser.add_argument('--multi', action='store_true', help='usar multitarea (solo para modelo fusion)')
    
    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Carga el conjunto de prueba
    test_anno = args.data_root + 'Training_Protocols/' + args.test_file
    test_dataset = AudioVisualDataset(test_anno, args.audio_path, args.visual_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            collate_fn=af_collate_fn, num_workers=4)
    
    # Carga el modelo
    model = load_model(args, args.model_path)
    print(f"Modelo cargado desde {args.model_path}")
    
    # Evalúa el modelo
    acc, f1, auc_score, report = evaluate_model(args, test_loader, model)
    
    # Guarda los resultados en un archivo
    result_file = os.path.join("results", f"eval_results_{os.path.basename(args.test_file)}.txt")
    os.makedirs("results", exist_ok=True)
    with open(result_file, 'w') as f:
        f.write(f"Test File: {args.test_file}\n")
        f.write(f"Model: {args.model_to_train}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"AUC: {auc_score:.4f}\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"Resultados guardados en {result_file}")

if __name__ == "__main__":
    main()