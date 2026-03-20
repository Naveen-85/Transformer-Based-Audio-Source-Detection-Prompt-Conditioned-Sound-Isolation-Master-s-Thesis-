import torch
import torch.nn as nn
import librosa
import numpy as np

SONYC_CLASSES = [
    "engine",
    "machinery_impact",
    "non_machinery_impact",
    "powered_saw",
    "alert_signal",
    "music",
    "human_voice",
    "dog",
]


# --------------------------
# Model
# --------------------------
class RobustAudioTransformerCNN(nn.Module):
    def __init__(
        self,
        num_classes=8,
        n_mels=128,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=512,
        dropout=0.2
    ):
        super().__init__()

        self.cnn_frontend = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8))
        )

        self.positional_encoding = nn.Parameter(torch.zeros(1, 32, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.attention_pool = nn.MultiheadAttention(
            d_model,
            4,
            dropout=dropout,
            batch_first=True
        )

        self.pool_queries = nn.Parameter(torch.randn(1, 4, d_model))

        self.output_layer = nn.Sequential(
            nn.LayerNorm(d_model * 5),
            nn.Linear(d_model * 5, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, num_classes)
        )

    def forward(self, x):
        b = x.size(0)

        x = self.cnn_frontend(x.unsqueeze(1))        # [B, D, 4, 8]
        x = x.view(b, x.size(1), -1).permute(0, 2, 1)  # [B, 32, D]

        x = x + self.positional_encoding
        x = self.transformer_encoder(x)

        q = self.pool_queries.expand(b, -1, -1)
        a, _ = self.attention_pool(q, x, x)          # [B, 4, D]

        g = x.mean(dim=1)                            # [B, D]

        return self.output_layer(torch.cat([a.reshape(b, -1), g], dim=1))


# --------------------------
# Audio preprocessing
# --------------------------
def audio_to_mel_10s(path: str, sr=16000):
    audio, _ = librosa.load(path, sr=sr, mono=True)

    target_len = sr * 10
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128,
        n_fft=1024,
        hop_length=160,
        win_length=400
    )

    mel = librosa.power_to_db(mel + 1e-10).astype(np.float32)

    if mel.shape[1] < 63:
        mel = np.pad(mel, ((0, 0), (0, 63 - mel.shape[1])))
    else:
        mel = mel[:, :63]

    return torch.tensor(mel).unsqueeze(0)   # [1, 128, 63]


# --------------------------
# Detector wrapper
# --------------------------
class SonycDetector:
    def __init__(self, ckpt_path: str, device: str):
        self.device = device
        self.model = RobustAudioTransformerCNN(num_classes=8).to(device)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        else:
            self.model.load_state_dict(ckpt, strict=True)

        self.model.eval()

    @torch.no_grad()
    def predict(self, audio_path: str, thresholds: dict):
        """
        Predict SONYC probabilities using thresholds passed from app.py
        """

        mel = audio_to_mel_10s(audio_path).to(self.device)
        probs = torch.sigmoid(self.model(mel))[0].cpu().numpy()

        results = []
        for cls, p in zip(SONYC_CLASSES, probs):

            thr = thresholds.get(cls, 0.5)

            results.append({
                "class": cls,
                "prob": float(p),
                "thr": float(thr),
                "yes": bool(p >= thr)
            })

        return results
