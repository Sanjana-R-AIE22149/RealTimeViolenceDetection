import torch
import torch.nn as nn
import torchvision.models as models


class ViolenceTransformerModel(nn.Module):
    def __init__(self, cnn_backbone='resnet18', embed_dim=512, num_layers=4, num_heads=8, dropout=0.1):
        super(ViolenceTransformerModel, self).__init__()

        # 1. Pretrained CNN backbone for feature extraction
        self.backbone = getattr(models, cnn_backbone)(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()  # remove final classification layer, output size 512 for resnet18

        # 2. Linear layer to embed CNN + motion features into transformer input
        self.embedding = nn.Linear(640,512)  

        # 3. Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)

        # 4. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, frame_batch, motion_batch):
     
        B, T, C, H, W = frame_batch.shape
        features = []

        for t in range(T):
            frame = frame_batch[:, t]  
            motion = motion_batch[:, t]  
            cnn_feat = self.backbone(frame)  
            feat = torch.cat([cnn_feat, motion], dim=1)  
            features.append(self.embedding(feat))  

        features = torch.stack(features, dim=1) 
        x = self.pos_encoder(features)
        x = self.transformer(x)  
        cls_token = x[:, -1, :]  
        logits = self.classifier(cls_token).squeeze(1)  # (B,)

        val=(logits+1) /2
        return val


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

     
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)  
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
