import torch
import torch.nn as nn
import torchaudio
from transformers import WavLMModel, ViTMAEModel
from torch.nn import functional as F

class ImprovedCrossFusionModule(nn.Module):
    """
    Enhanced crossmodal fusion module with dynamic modality weighting
    """
    def __init__(self, dim=256):
        super(ImprovedCrossFusionModule, self).__init__()
        # linear project + norm + corr + concat + conv_layer + tanh
        self.project_audio = nn.Linear(768, dim)  # linear projection
        self.project_vision = nn.Linear(768, dim)
        self.corr_weights = torch.nn.Parameter(torch.empty(
            dim, dim, requires_grad=True).type(torch.cuda.FloatTensor))
        nn.init.xavier_normal_(self.corr_weights)
        self.project_bottleneck = nn.Sequential(nn.Linear(dim * 2, 64),
                                                nn.LayerNorm((64,), eps=1e-05, elementwise_affine=True),
                                                nn.ReLU())
    
    def forward(self, audio_feat, visual_feat):
        """

        :param audio_feat: [batchsize 64 768]
        :param visual_feat:[batchsize 64 768]
        :return: fused feature
        """
        audio_feat = self.project_audio(audio_feat)
        visual_feat = self.project_vision(visual_feat)

        # Compute correlation matrix
        a1 = torch.matmul(audio_feat, self.corr_weights)  # [batchsize, dim]
        cc_mat = torch.matmul(a1, visual_feat.transpose(0, 1))  # [batchsize, batchsize]
        
        # Compute attention weights
        audio_att = F.softmax(cc_mat, dim=1)  # [batchsize, batchsize]
        visual_att = F.softmax(cc_mat.transpose(0, 1), dim=1)  # [batchsize, batchsize]
        
        # Apply attention
        atten_audiofeatures = torch.matmul(audio_att, audio_feat)  # [batchsize, dim]
        atten_visualfeatures = torch.matmul(visual_att, visual_feat)  # [batchsize, dim]
        
        # Residual connection
        atten_audiofeatures = atten_audiofeatures + audio_feat
        atten_visualfeatures = atten_visualfeatures + visual_feat
        
        # Concatenate and project
        fused_features = self.project_bottleneck(torch.cat((atten_audiofeatures,
                                                          atten_visualfeatures), dim=1))
        
        return fused_features

class EnhancedDeceptionDetector(nn.Module):
    """
    Improved deception detection model with:
    - MAE-ViT for visual features
    - WavLM for audio features
    - Enhanced fusion mechanism
    """
    def __init__(self, num_encoders=4, use_adapter=True):
        super(EnhancedDeceptionDetector, self).__init__()
        
        # Audio encoder (WavLM)
        self.audio_encoder = WavLMModel.from_pretrained("microsoft/wavlm-base")
        for param in self.audio_encoder.parameters():
            param.requires_grad = False  # Freeze pretrained model
            
        for layer in self.audio_encoder.encoder.layers[-2:]:
          for param in layer.parameters():
            param.requires_grad = True
            
        # Audio adapter layers
        self.audio_adapter = nn.ModuleList([
            nn.Sequential(
                nn.Linear(768, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, 768))
            for _ in range(num_encoders)])
        
        # Visual encoder (MAE-ViT)
        self.visual_encoder = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
        for param in self.visual_encoder.parameters():
            param.requires_grad = False  # Freeze pretrained model
        for block in self.visual_encoder.encoder.layer[-2:]:
            for param in block.parameters():
                param.requires_grad = True
            
        # Visual adapter layers
        self.visual_adapter = nn.ModuleList([
            nn.Sequential(
                nn.Linear(768, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, 768))
            for _ in range(num_encoders)])
        # Fusion modules
        self.fusion_layers = nn.ModuleList([
            ImprovedCrossFusionModule(dim=512)
            for _ in range(num_encoders)])
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 * num_encoders, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2))
        
        # Positional embeddings for visual tokens
        self.visual_pos_embed = nn.Parameter(torch.randn(1, 50, 768))
        
        nn.init.trunc_normal_(self.visual_pos_embed, std=0.02)
        
        # Capa de atención temporal para audio
        self.audio_temp_attention = nn.MultiheadAttention(768, 8, dropout=0.1)
        
        # Regularización
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, audio_input, visual_input):
        # Audio processing con atención temporal
        audio_outputs = self.audio_encoder(audio_input)
        audio_features = audio_outputs.last_hidden_state  # (batch_size, seq_len, 768)
        audio_features = audio_features.permute(1, 0, 2)  # (seq_len, batch_size, 768)
        audio_features, _ = self.audio_temp_attention(audio_features, audio_features, audio_features)
        audio_features = audio_features.permute(1, 0, 2)  # (batch_size, seq_len, 768)
        
        # Visual processing
        b, t, c, h, w = visual_input.shape
        visual_input = visual_input.view(b * t, c, h, w)
        visual_outputs = self.visual_encoder(visual_input)
        visual_features = visual_outputs.last_hidden_state  # (b*t, seq_len, 768)
        visual_features = visual_features.view(b, t, -1, 768)  # (b, t, seq_len, 768)
        visual_features = visual_features + self.visual_pos_embed.unsqueeze(1)
        
        # Procesamiento multimodal
        fusion_outputs = []
        for i in range(len(self.fusion_layers)):
            # Atención temporal en lugar de mean pooling
            audio_pooled = audio_features.mean(dim=1)  # Temporalmente, mantener mean pooling
            audio_adapted = self.audio_adapter[i](self.dropout(audio_pooled))
            
            visual_pooled = visual_features.mean(dim=(1, 2))
            visual_adapted = self.visual_adapter[i](self.dropout(visual_pooled))
            
            # Fusión
            fused = self.fusion_layers[i](audio_adapted, visual_adapted)
            fusion_outputs.append(fused)
        
        combined = torch.cat(fusion_outputs, dim=1)  # Asegurar concatenación correcta
        
        # Clasificación
        logits = self.classifier(combined)
        
        return logits, None, None

# Example usage
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = EnhancedDeceptionDetector(num_encoders=4).to(device)
    
    # Test inputs
    audio_sample = torch.randn(2, 16000).to(device)  # Batch of 2 audio clips
    video_sample = torch.randn(2, 64, 3, 160, 160).to(device)  # Batch of 2 video clips (64 frames each)
    
    # Forward pass
    outputs = model(audio_sample, video_sample)
    print(f"Output logits shape: {outputs.shape}")  # Should be [2, 2]