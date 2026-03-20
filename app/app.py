import os
import tempfile
import streamlit as st
import numpy as np
import torch

from audio_io import (
    load_audio_mono,
    pad_or_trim,
    peak_normalize,
    to_torch_1x1xT,
    save_wav
)

from sonyc_detector import SonycDetector, SONYC_CLASSES
from flow_separator import FlowSeparator, SR as SEP_SR, DATASET_LEN


# --------------------------
# Page setup
# --------------------------
st.set_page_config(page_title="Selective Hearing", layout="centered")
st.title("🎧 Transformer Detection + Diffusion Selective Hearing")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
st.caption(f"Device: **{DEVICE.upper()}**")


# --------------------------
# Paths
# --------------------------
SONYC_CKPT = st.text_input("SONYC checkpoint", "../checkpoints/student_no_kd_best_511.pt")
FLOW_CKPT  = st.text_input("Flow separator checkpoint", "../checkpoints/best.pt")


# --------------------------
# Thresholds (EDIT HERE)
# --------------------------
THRESHOLDS = {
    "engine": 0.80,
    "machinery_impact": 0.90,
    "non_machinery_impact": 0.50,
    "powered_saw": 0.50,
    "alert_signal": 0.75,
    "music": 0.015,
    "human_voice": 0.20,
    "dog": 0.012,
}

st.subheader("⚙️ Current Detection Thresholds")

for cls in THRESHOLDS:
    st.write(f"**{cls}** → {THRESHOLDS[cls]}")


# --------------------------
# Cache models
# --------------------------
@st.cache_resource
def load_detector(ckpt, device):
    return SonycDetector(ckpt, device)

@st.cache_resource
def load_separator(ckpt, device):
    return FlowSeparator(ckpt, device)


# --------------------------
# Check checkpoints exist
# --------------------------
if not os.path.exists(SONYC_CKPT):
    st.error(f"❌ Missing SONYC checkpoint: {SONYC_CKPT}")
    st.stop()

if not os.path.exists(FLOW_CKPT):
    st.error(f"❌ Missing Flow checkpoint: {FLOW_CKPT}")
    st.stop()


detector = load_detector(SONYC_CKPT, DEVICE)
separator = load_separator(FLOW_CKPT, DEVICE)


# --------------------------
# Upload audio
# --------------------------
st.subheader("📤 Upload mixture audio")
uploaded = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])

if uploaded is None:
    st.stop()


tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
tmp.write(uploaded.read())
tmp.flush()
audio_path = tmp.name


st.subheader("🎵 Original audio")
st.audio(audio_path)


# --------------------------
# Step 1 — Detection
# --------------------------
st.subheader("Step 1 — Sound detection (SONYC)")

if st.button("🚀 Run detection"):

    # Pass thresholds from app.py into detector
    det = detector.predict(audio_path, THRESHOLDS)

    st.session_state["det_results"] = det


det_results = st.session_state.get("det_results", None)

if det_results:
    st.success("Detection completed ✅")

    st.markdown("### 📊 Detection Results")

    for r in det_results:
        tag = "✅ YES" if r["yes"] else "❌ NO"
        st.write(
            f"**{r['class']}**: {r['prob']:.3f} "
            f"(thr={r['thr']:.3f}) → {tag}"
        )

    yes_classes = [r["class"] for r in det_results if r["yes"]]

    if yes_classes:
        st.markdown("### ✅ Detected YES classes")
        st.write(", ".join([f"**{c}**" for c in yes_classes]))
    else:
        st.warning("No class exceeded thresholds.")


# --------------------------
# Step 2 — Isolation
# --------------------------
st.subheader("Step 2 — Selective hearing (prompt isolation)")

if not det_results:
    st.info("Run detection first to enable isolation.")
    st.stop()


yes_classes = [r["class"] for r in det_results if r["yes"]]

if len(yes_classes) == 0:
    st.warning("Nothing detected above threshold → cannot isolate.")
    st.stop()


# --------------------------
# Prompt mapping (trained prompts only)
# --------------------------
PROMPT_MAP = {
    "dog": "dog barking",
    "human_voice": "human speech",
    "music": "music",
}


detected_prompts = []
for c in yes_classes:
    if c in PROMPT_MAP:
        detected_prompts.append(PROMPT_MAP[c])

detected_prompts = list(dict.fromkeys(detected_prompts))

if len(detected_prompts) == 0:
    st.error("❌ No detected class matches trained prompts.")
    st.stop()


st.success("Available trained prompts: " + ", ".join(detected_prompts))


prompt = st.selectbox("🎯 Choose what to isolate", detected_prompts)

steps = st.slider("Inference steps", 60, 250, 120, 10)


# --------------------------
# Run Isolation
# --------------------------
if st.button("🎛️ Run isolation"):
    with st.spinner("Isolating sound... please wait"):

        y = load_audio_mono(audio_path, sr=SEP_SR)
        y = pad_or_trim(y, DATASET_LEN)
        y = peak_normalize(y, 0.95)

        mix_t = to_torch_1x1xT(y, DEVICE)

        out = separator.isolate(mix_t, prompt=prompt, steps=steps)
        out = peak_normalize(out, 0.95)

        out_path = os.path.join(tempfile.gettempdir(), "isolated.wav")
        save_wav(out_path, out, SEP_SR)

        st.session_state["isolated_path"] = out_path
        st.session_state["isolated_prompt"] = prompt


# --------------------------
# Output
# --------------------------
isolated_path = st.session_state.get("isolated_path", None)

if isolated_path and os.path.exists(isolated_path):

    st.subheader("✅ Isolated output")
    st.write("Prompt:", st.session_state.get("isolated_prompt", ""))

    st.audio(isolated_path)

    with open(isolated_path, "rb") as f:
        st.download_button(
            "⬇️ Download isolated.wav",
            f,
            file_name="isolated.wav",
            mime="audio/wav"
        )
