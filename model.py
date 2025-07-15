import torch
import torch.nn as nn

class SequenceClassifier(nn.Module):
    def __init__(self, input_dim, max_seq_len=512, d_model=64, nhead=8, num_layers=6, dim_feedforward=64, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(nn.LayerNorm(input_dim),
                                        nn.Linear(input_dim, d_model))
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.input_proj(x) + self.position_embedding(pos)
        features = self.transformer(x).mean(dim=1)
        return self.classifier(features)

if __name__ == "__main__":
    model = SequenceClassifier(input_dim=128)
    dummy_input = torch.randn(32, 50, 128)
    output = model(dummy_input)
    print(output.shape)