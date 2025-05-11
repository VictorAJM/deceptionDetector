import torch
import torch.nn as nn

class MultimodalLieDetector(nn.Module):
    def __init__(self, video_model, audio_model, dropout_prob=0.5):
        super().__init__()
        self.video_model = video_model
        self.audio_model = audio_model
        
        # Ensure both models have a config attribute
        if not hasattr(self.video_model, 'config') or not hasattr(self.audio_model, 'config'):
            raise ValueError("Both video_model and audio_model must have a 'config' attribute.")
        
        # Combine the hidden sizes from both models
        combined_hidden_size = self.video_model.config.hidden_size + self.audio_model.config.hidden_size
        
        self.dropout = nn.Dropout(dropout_prob)  # Añadido Dropout
        self.classifier = nn.Linear(video_model.config.hidden_size + audio_model.config.hidden_size, 2)

    def forward(self, video_inputs, audio_inputs):
        # Get video features
        video_outputs = self.video_model(video_inputs)
        video_features = video_outputs.hidden_states[-1][:, 0, :]  # Only if hidden_states enabled

        # Get audio features
        audio_outputs = self.audio_model(audio_inputs)
        audio_features = audio_outputs  # Now shape [B, H], no extra indexing needed


        combined_features = torch.cat((video_features, audio_features), dim=1)
        combined_features = self.dropout(combined_features)
        logits = self.classifier(combined_features)
        return logits
