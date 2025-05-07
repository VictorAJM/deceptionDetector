import torch
import torch.nn as nn

class MultimodalLieDetector(nn.Module):
    def __init__(self, video_model, audio_model):
        super(MultimodalLieDetector, self).__init__()
        self.video_model = video_model
        self.audio_model = audio_model
        self.classifier = nn.Linear(video_model.config.hidden_size + audio_model.config.hidden_size, 2)

    def forward(self, video_inputs, audio_inputs):
        video_outputs = self.video_model(video_inputs)
        audio_outputs = self.audio_model(audio_inputs)

        combined_features = torch.cat((video_outputs.last_hidden_state[:, 0, :], audio_outputs.last_hidden_state[:, 0, :]), dim=1)
        logits = self.classifier(combined_features)
        return logits