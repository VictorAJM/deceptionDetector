import torch
import torch.nn as nn

################# Non-Linear Adapters ######################

class w2v2_adapter_nlp(nn.Module):
    def __init__(self, transformer_encoder):
        super(w2v2_adapter_nlp, self).__init__()

        self.attention = transformer_encoder.attention
        self.dropout = transformer_encoder.dropout
        self.layer_norm = transformer_encoder.layer_norm

        self.adapter = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 768),
            nn.LayerNorm(768),
        )
        self.feed_forward = transformer_encoder.feed_forward
        self.final_layer_norm = transformer_encoder.final_layer_norm

    def forward(self, x):
        mhsa = self.attention(x)[0] + x  # Evita error de múltiples valores
        mhsa = self.dropout(mhsa)
        mhsa = self.layer_norm(mhsa)
        adapter_seq = self.adapter(mhsa) + mhsa
        ffn = self.feed_forward(adapter_seq) + adapter_seq
        return self.final_layer_norm(ffn)


class vit_adapter_nlp(nn.Module):
    def __init__(self, transformer_encoder):
        super(vit_adapter_nlp, self).__init__()

        self.ln1 = transformer_encoder.ln_1
        self.attention = transformer_encoder.self_attention
        self.drop = transformer_encoder.dropout

        self.adapter = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 768),
            nn.LayerNorm(768),
        )
        self.ln2 = transformer_encoder.ln_2
        self.mlp = transformer_encoder.mlp

    def forward(self, x):
        norm_x = self.ln1(x)
        attention, _ = self.attention(norm_x, norm_x, norm_x, need_weights=False)
        mhsa = self.drop(attention) + x
        adapter_seq = self.adapter(mhsa) + mhsa
        ffn = self.mlp(adapter_seq) + adapter_seq
        return self.ln2(ffn)


################# Convolutional Adapters ####################

class Efficient_Conv_Pass(nn.Module):
    def __init__(self):
        super(Efficient_Conv_Pass, self).__init__()
        self.adapter_down = nn.Linear(768, 32)
        self.adapter_gelu = nn.GELU()
        self.adapter_1d_cnn = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.adapter_up = nn.Linear(32, 768)
        self.adapter_norm = nn.LayerNorm(768)

    def forward(self, x):
        down = self.adapter_gelu(self.adapter_down(x))
        down = down.permute(0, 2, 1)
        conv = self.adapter_gelu(self.adapter_1d_cnn(down))
        conv = conv.permute(0, 2, 1)
        up = self.adapter_gelu(self.adapter_up(conv))
        return self.adapter_norm(up + x)


class w2v2_adapter_conv(nn.Module):
    def __init__(self, transformer_encoder):
        super(w2v2_adapter_conv, self).__init__()
        self.attention = transformer_encoder.attention
        self.dropout = transformer_encoder.dropout
        self.layer_norm = transformer_encoder.layer_norm
        self.mhsa_conv_pass = Efficient_Conv_Pass()
        self.ffn_conv_pass = Efficient_Conv_Pass()
        self.adapter_norm1 = nn.LayerNorm(768)
        self.adapter_norm2 = nn.LayerNorm(768)
        self.feed_forward = transformer_encoder.feed_forward
        self.final_layer_norm = transformer_encoder.final_layer_norm

    def forward(self, x):
        mhsa = self.attention(x)[0] + x + self.mhsa_conv_pass(x)
        mhsa = self.adapter_norm1(mhsa)
        ffn = self.feed_forward(mhsa) + mhsa + self.ffn_conv_pass(mhsa)
        return self.adapter_norm2(ffn)


class vit_adapter_conv(nn.Module):
    def __init__(self, transformer_encoder):
        super(vit_adapter_conv, self).__init__()
        self.ln1 = transformer_encoder.ln_1
        self.attention = transformer_encoder.self_attention
        self.drop = transformer_encoder.dropout
        self.mhsa_conv_pass = Efficient_Conv_Pass()
        self.ffn_conv_pass = Efficient_Conv_Pass()
        self.adapter_norm1 = nn.LayerNorm(768)
        self.adapter_norm2 = nn.LayerNorm(768)
        self.ln2 = transformer_encoder.ln_2
        self.mlp = transformer_encoder.mlp

    def forward(self, x):
        conv_pass = self.mhsa_conv_pass(x)
        norm_x = self.ln1(x)
        attention, _ = self.attention(norm_x, norm_x, norm_x, need_weights=False)
        attention = self.drop(attention)
        mhsa = attention + x + conv_pass
        mhsa = self.adapter_norm1(mhsa)
        ffn = self.mlp(mhsa) + mhsa + self.ffn_conv_pass(mhsa)
        return self.adapter_norm2(ffn)
