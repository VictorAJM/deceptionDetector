import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalLieDetectorv2(nn.Module):
    def __init__(self, video_model, audio_model, dropout_prob=0.5):
        super().__init__()
        self.video_model = video_model
        self.audio_model = audio_model
        
        # Verificar que ambos modelos tengan 'config'
        if not hasattr(self.video_model, 'config') or not hasattr(self.audio_model, 'config'):
            raise ValueError("Both video_model and audio_model must have a 'config' attribute.")
        
        # Dimensiones de los embeddings
        H_v = self.video_model.config.hidden_size
        H_a = self.audio_model.config.hidden_size
        H_comb = H_v + H_a
        
        # Cabeza de fusión: MLP de dos capas con activación y dropout
        hidden_dim = H_comb // 2
        self.fusion_mlp = nn.Sequential(
            nn.Linear(H_comb, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
        )
        
        # Clasificador final
        self.classifier = nn.Linear(hidden_dim // 2, 2)

    def forward(self, video_inputs, audio_inputs):
        # Extraer features de video
        vid_out = self.video_model(video_inputs)
        v = vid_out.hidden_states[-1][:, 0, :]   # [B, H_v]
        
        # Extraer features de audio
        a = self.audio_model(audio_inputs)       # [B, H_a]
        
        # Fusionar y pasar por MLP
        combined = torch.cat([v, a], dim=1)      # [B, H_v+H_a]
        x = self.fusion_mlp(combined)            # [B, hidden_dim//2]
        
        # Logits finales
        logits = self.classifier(x)              # [B, 2]
        return logits
