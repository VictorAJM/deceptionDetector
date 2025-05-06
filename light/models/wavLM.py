import torch.nn as nn
from transformers import WavLMModel

class WavLMLieDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base")
        # Freeze WavLM parameters if you want to only train the classifier
        # for param in self.wavlm.parameters():
        #     param.requires_grad = False
            
        self.classifier = nn.Sequential(
            nn.Linear(self.wavlm.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # Binary classification
        )

    def forward(self, input_values):
        # Expected input shape: [batch_size, sequence_length]
        
        # WavLM expects input shape [batch_size, sequence_length]
        outputs = self.wavlm(input_values)
        
        # Get last hidden states [batch_size, seq_len, hidden_size]
        hidden_states = outputs.last_hidden_state
        
        # Mean pooling over time dimension
        pooled = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        
        # Classifier
        logits = self.classifier(pooled)  # [batch_size, 2]
        return logits