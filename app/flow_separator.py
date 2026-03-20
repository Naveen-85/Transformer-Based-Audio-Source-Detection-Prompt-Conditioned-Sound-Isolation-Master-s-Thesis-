import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer, T5EncoderModel
from encodec import EncodecModel


SR = 24000
DATASET_SECONDS = 6
DATASET_LEN = SR * DATASET_SECONDS

D_MODEL = 192
N_HEADS = 3
N_LAYERS = 4

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.sa = nn.MultiheadAttention(d, N_HEADS, batch_first=True)
        self.ca = nn.MultiheadAttention(d, N_HEADS, batch_first=True)
        self.tp = nn.Linear(768, d)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.n3 = nn.LayerNorm(d)

    def forward(self, x, tok):
        x = x + self.sa(self.n1(x), self.n1(x), self.n1(x))[0]
        tt = self.tp(tok)
        x = x + self.ca(self.n2(x), tt, tt)[0]
        return x + self.ff(self.n3(x))

class FlowModel(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.inp = nn.Linear(2*C, D_MODEL)
        self.blocks = nn.ModuleList([Block(D_MODEL) for _ in range(N_LAYERS)])
        self.out = nn.Linear(D_MODEL, C)

    def forward(self, z_t, z_mix, t, txt):
        x = torch.cat([z_t, z_mix], 1).permute(0, 2, 1)  # [B,T,2C]
        x = self.inp(x)
        for b in self.blocks:
            x = b(x, txt)
        return self.out(x).permute(0, 2, 1)  # [B,C,T]


def crop_or_pad_torch(wav: torch.Tensor, T=DATASET_LEN):
    if wav.shape[-1] >= T:
        return wav[..., :T]
    return F.pad(wav, (0, T - wav.shape[-1]))


class FlowSeparator:
    """
    Loads:
      - T5 encoder (text conditioning)
      - EnCodec (latent codec)
      - FlowModel checkpoint (best.pt)
    Provides:
      isolate(mix_tensor_1x1xT, prompt) -> np audio
    """
    def __init__(self, ckpt_path: str, device: str):
        self.device = device

        # text
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.text_enc = T5EncoderModel.from_pretrained("t5-base").to(device).eval()
        for p in self.text_enc.parameters():
            p.requires_grad = False

        # codec
        self.codec = EncodecModel.encodec_model_24khz().to(device).eval()
        self.codec.set_target_bandwidth(6.0)
        for p in self.codec.parameters():
            p.requires_grad = False

        self._model = None
        self._ckpt_path = ckpt_path

    @torch.no_grad()
    def encode_text(self, prompts):
        toks = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        return self.text_enc(**toks).last_hidden_state  # [B,L,768]

    @torch.no_grad()
    def audio_encode(self, w):
        return self.codec.encoder(w)

    @torch.no_grad()
    def audio_decode(self, z):
        return self.codec.decoder(z)

    def _ensure_model_loaded(self, C: int):
        if self._model is not None:
            return
        self._model = FlowModel(C).to(self.device)
        state = torch.load(self._ckpt_path, map_location=self.device)
        self._model.load_state_dict(state, strict=False)
        self._model.eval()

    @torch.no_grad()
    def isolate(self, mix_1x1xT: torch.Tensor, prompt: str, steps: int = 120):
        """
        mix_1x1xT: torch tensor [1,1,T] at SR=24000
        """
        mix_1x1xT = crop_or_pad_torch(mix_1x1xT).to(self.device)

        zm = self.audio_encode(mix_1x1xT)  # [1,C,Tz]
        C = zm.shape[1]
        self._ensure_model_loaded(C)

        txt = self.encode_text([prompt])   # [1,L,768]

        z = torch.randn_like(zm)
        dt = -1.0 / (steps - 1)

        ts = torch.linspace(1.0, 0.0, steps, device=self.device)
        for t in ts:
            tt = torch.tensor([float(t)], device=self.device)
            v = self._model(z, zm, tt, txt)
            z = z + v * dt

        wav = self.audio_decode(z)[0, 0].clamp(-1, 1).detach().cpu().numpy()
        return wav
