# Transformer Based Selective Hearing System for Real-Time Audio Filtering and Source Separation

## Overview

This project presents a **selective hearing system** for audio analysis, filtering, and source separation. The system is designed to identify important sound events in a mixed audio signal and isolate the target sound of interest from the mixture. It combines **sound event classification** with **prompt-conditioned sound isolation** in a unified pipeline. 

The application was developed as part of the thesis project **“Transformer Based Selective Hearing System for Real-Time Audio Filtering and Source Separation.”** The core idea is to simulate selective listening: first understand what sound is present, then separate the desired sound from overlapping background audio.

---

## What This Application Does

The application allows a user to:

- upload an audio mixture
- analyze the audio to detect the dominant sound event
- use the predicted sound label to guide the isolation process
- generate a cleaner output containing the target sound source

The proposed framework uses a **two-stage architecture**:

1. **Sound Event Classification**  
   The input audio is converted into log-mel spectrogram features and classified using a knowledge-distilled model. A large **Audio Spectrogram Transformer (AST)** model acts as the teacher, and a smaller CNN-Transformer student model is used for efficient inference.

2. **Prompt-Based Sound Isolation**  
   The predicted sound class is then used as a conditioning signal for the isolation model. The model isolates the target sound from the mixed audio using a prompt-guided generative approach. 

This makes the system suitable for applications such as assistive listening, smart audio systems, environmental sound understanding, and audio source separation research. 

---

## Key Features

- Two-stage selective hearing pipeline
- Sound event classification using knowledge distillation
- Prompt-conditioned sound isolation
- Interactive Streamlit-based user interface
- Upload-and-test workflow for audio mixtures
- Research-oriented demonstration of real-time audio filtering and source separation

---

## Project Workflow

The system follows this general pipeline:

**Input audio mixture → preprocessing → sound classification → predicted sound label → prompt generation → sound isolation → output audio**

In practice, the uploaded audio is processed as follows:

- audio is standardized and transformed into features for classification
- the classification model predicts the target sound event
- the predicted class is used as a prompt for the isolation stage
- the isolation model reconstructs the desired sound source
- the application returns the processed output to the user 

---

## Requirements

Make sure Python is installed on your system.

Install the required dependencies using:pip install -r requirements.txt

How to Run the Application

Start the Streamlit app with: streamlit run app.py

```bash
pip install -r requirements.txt
