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
        
        # Projections with layer normalization
        self.project_audio = nn.Sequential(
            nn.Linear(1024, dim),
            nn.LayerNorm(dim)
        )
        self.project_vision = nn.Sequential(
            nn.Linear(768, dim),
            nn.LayerNorm(dim))
        
        # Attention mechanisms
        self.audio_attention = nn.MultiheadAttention(dim, num_heads=4)
        self.visual_attention = nn.MultiheadAttention(dim, num_heads=4)
        
        # Dynamic modality weighting
        self.modality_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(0.1))
    
    def forward(self, audio_feat, visual_feat):
        # Project features to common space
        audio_proj = self.project_audio(audio_feat)
        visual_proj = self.project_vision(visual_feat)
        
        # Cross-modal attention
        audio_attn, _ = self.audio_attention(
            audio_proj.transpose(0, 1), 
            visual_proj.transpose(0, 1), 
            visual_proj.transpose(0, 1))
        visual_attn, _ = self.visual_attention(
            visual_proj.transpose(0, 1),
            audio_proj.transpose(0, 1),
            audio_proj.transpose(0, 1))
        
        audio_attn = audio_attn.transpose(0, 1)
        visual_attn = visual_attn.transpose(0, 1)
        
        # Residual connections
        audio_out = audio_proj + audio_attn
        visual_out = visual_proj + visual_attn
        
        # Dynamic modality weighting
        weights = F.softmax(self.modality_weights, dim=0)
        fused = torch.cat([
            weights[0] * audio_out,
            weights[1] * visual_out
        ], dim=-1)
        
        return self.bottleneck(fused)

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
            
        # Audio adapter layers
        self.audio_adapter = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, 256),
                nn.GELU(),
                nn.Linear(256, 1024))
            for _ in range(num_encoders)])
        
        # Visual encoder (MAE-ViT)
        self.visual_encoder = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
        for param in self.visual_encoder.parameters():
            param.requires_grad = False  # Freeze pretrained model
            
        # Visual adapter layers
        self.visual_adapter = nn.ModuleList([
            nn.Sequential(
                nn.Linear(768, 256),
                nn.GELU(),
                nn.Linear(256, 768))
            for _ in range(num_encoders)])
        
        # Fusion modules
        self.fusion_layers = nn.ModuleList([
            ImprovedCrossFusionModule(dim=256)
            for _ in range(num_encoders)])
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 * num_encoders, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2))
        
        # Positional embeddings for visual tokens
        self.visual_pos_embed = nn.Parameter(torch.randn(1, 64, 768))
        
    def forward(self, audio_input, visual_input):
        # Audio processing
        audio_features = self.audio_encoder(audio_input).last_hidden_state
        
        # Visual processing
        b, t, c, h, w = visual_input.shape
        visual_input = visual_input.view(b * t, c, h, w)
        visual_features = self.visual_encoder(visual_input).last_hidden_state
        visual_features = visual_features.view(b, t, -1) + self.visual_pos_embed
        
        # Adapter processing and fusion
        fusion_outputs = []
        for i in range(len(self.fusion_layers)):
            # Apply adapters
            audio_adapted = self.audio_adapter[i](audio_features)
            visual_adapted = self.visual_adapter[i](visual_features)
            
            # Fusion
            fused = self.fusion_layers[i](audio_adapted, visual_adapted)
            fusion_outputs.append(fused)
        
        # Combine fusion outputs
        combined = torch.cat(fusion_outputs, dim=-1)
        
        # Classification
        logits = self.classifier(combined.mean(dim=1))
        
        return logits

# Example usage
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = EnhancedDeceptionDetector(num_encoders=4).to(device)
    
    # Test inputs
    audio_sample = torch.randn(2, 16000).to(device)  # Batch of 2 audio clips
    video_sample = torch.randn(2, 64, 3, 224, 224).to(device)  # Batch of 2 video clips (64 frames each)
    
    # Forward pass
    outputs = model(audio_sample, video_sample)
    print(f"Output logits shape: {outputs.shape}")  # Should be [2, 2]