import numpy as np
import librosa
import soundfile as sf

def load_audio_mono(path: str, sr: int):
    y, sr0 = librosa.load(path, sr=None, mono=True)
    if sr0 != sr:
        y = librosa.resample(y, orig_sr=sr0, target_sr=sr)
    y = np.clip(y, -1.0, 1.0)
    return y.astype(np.float32)

def pad_or_trim(y: np.ndarray, target_len: int):
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)))
    return y[:target_len]

def peak_normalize(y: np.ndarray, peak: float = 0.95):
    m = np.max(np.abs(y)) + 1e-8
    return (y / m * peak).astype(np.float32)

def save_wav(path: str, y: np.ndarray, sr: int):
    sf.write(path, y, sr)

def to_torch_1x1xT(y: np.ndarray, device: str):
    import torch
    t = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)  # [1,1,T]
    return t.to(device)
