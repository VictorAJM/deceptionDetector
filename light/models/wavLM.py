import torch.nn as nn
from transformers import WavLMModel

class WavLMLieDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base")
        self.config = self.wavlm.config

    def forward(self, input_values):
        outputs = self.wavlm(input_values)
        hidden_states = outputs.last_hidden_state  # [B, T, H]
        pooled = hidden_states.mean(dim=1)         # [B, H]
        return pooled  # ✔ Return features only, not logits
