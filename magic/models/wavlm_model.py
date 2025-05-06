import torch
import torch.nn as nn
import torchaudio
from models.adapter import wavlm_adapter_nlp, wavlm_adapter_conv

class WavLM_Model(nn.Module):
    def __init__(self, num_encoders, adapter, adapter_type):
        super(WavLM_Model, self).__init__()
        
        self.num_encoders = num_encoders
        self.adapter = adapter
        self.adapter_type = adapter_type

        model = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
        # Congelar todos los parámetros inicialmente
        for param in model.parameters():
            param.requires_grad = False

        self.FEATURE_EXTRACTOR = model.feature_extractor
        self.FEATURE_PROJECTOR = nn.Sequential(
            model.encoder.feature_projection,
            model.encoder.transformer.pos_conv_embed,
            model.encoder.transformer.layer_norm,
            model.encoder.transformer.dropout,
        )

        layer_list = []
        self.encoders = nn.ModuleList()
        for i in range(self.num_encoders):

            if self.adapter:
                if self.adapter_type == 'nlp':
                    self.encoders.append(self._make_nlp_adapter(model.encoder.transformer.layers[i]))
                else:
                    self.encoders.append(self._make_conv_adapter(model.encoder.transformer.layers[i]))
            else:
                # fine_tune enoder in case we donot use adapters
                for p in model.encoder.transformer.layers[i].parameters(): p.requires_grad = True
                self.encoders.append(model.encoder.transformer.layers[i])

        self.TRANSFORMER = nn.Sequential(*layer_list)
        # Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(768, 2)
        )
    def _make_nlp_adapter(self, original_layer):
        # Adaptador estilo NLP (bottleneck)
        class Adapter(nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.layer = layer
                self.adapter = nn.Sequential(
                    nn.Linear(768, 64),
                    nn.GELU(),
                    nn.Linear(64, 768)
                )
                
            def forward(self, x):
                original_output = self.layer(x)
                if isinstance(original_output, tuple):
                    # Si la capa devuelve una tupla, aplicamos el adaptador solo al primer elemento
                    adapter_output = original_output[0] + self.adapter(original_output[0])
                    return (adapter_output,) + original_output[1:]
                else:
                    return original_output + self.adapter(original_output)
                
        return Adapter(original_layer)

    def _make_conv_adapter(self, original_layer):
        # Adaptador convolucional
        class Adapter(nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.layer = layer
                self.adapter = nn.Sequential(
                    nn.Conv1d(768, 64, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv1d(64, 768, kernel_size=1)
                )
                
            def forward(self, x):
                original_output = self.layer(x)
                # Reordenar dimensiones para Conv1d (batch, channels, time)
                if isinstance(original_output, tuple):
                    h = original_output[0].transpose(1, 2)
                    h = self.adapter(h).transpose(1, 2)
                    adapter_output = (original_output[0] + h,) + original_output[1:]
                    return adapter_output
                else:
                    h = original_output.transpose(1, 2)
                    h = self.adapter(h).transpose(1, 2)
                    return original_output + h
                
        return Adapter(original_layer)
    def forward(self, x):
        x, _ = self.FEATURE_EXTRACTOR(x, None)
        x = self.FEATURE_PROJECTOR(x)
        # Capas transformer seleccionadas
        for layer in self.encoders:
            x = layer(x)
            # Extraer el tensor si se devuelve una tupla
            if isinstance(x, tuple):
                x = x[0]

        logits = self.classifier(x)
        return torch.mean(logits, 1)