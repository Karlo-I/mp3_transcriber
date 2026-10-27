# Local MP3 Transcriber

A lightweight, 100% offline Flask web application that transcribes audio files into text using the open-source Whisper model. This tool is specifically optimized for Apple Silicon (M1/M2/M3) Macs, utilizing the Metal Performance Shaders (MPS) backend for blazing-fast local processing. The front-end is very basic, which is enough for my use-case, but feel free to pull-request to improve on it as you wish. 

## ✨ Features

*   **100% Offline & Private:** No audio data ever leaves your computer.
*   **Apple Silicon Optimized:** Runs on the Mac's GPU (MPS) for rapid transcription.
*   **Custom Output Paths:** Easily select where you want your transcripts saved.
*   **Timestamped Output:** Generates clean, readable text files with timecodes.

##  How Whisper Technology Works

Whisper does not "hear" audio like a human; it uses deep learning to translate sound waves into text. Here is the high-level process:

1.  **Audio to Numbers:** The `.mp3` file is decoded into raw numerical data representing the amplitude of sound waves over time.
2.  **Mel-Spectrogram Generation:** Whisper converts these raw waves into a **Mel-spectrogram**. This is a 2D visual heatmap where the X-axis is time, the Y-axis is frequency (pitch), and the color intensity represents volume. *This heatmap is what the AI actually "sees."*
3.  **Transformer Architecture:** The model uses an Encoder-Decoder architecture:
    *   **The Encoder:** Analyzes the spectrogram to understand acoustic patterns, phonetics, and context.
    *   **The Decoder:** Predicts the most likely sequence of text tokens (words) based on those patterns, generating the transcript piece by piece.
4.  **Training:** It knows how to map these heatmaps to text because it was trained on 680,000 hours of diverse, multilingual audio, allowing it to handle accents, background noise, and technical jargon effortlessly.

## ️ Prerequisites

*   Python 3.8+
*   FFmpeg (Required for processing `.mp3` files)
    *   *Install via Homebrew:* `brew install ffmpeg`

## 🚀 Getting Started

1.  **Clone or open the project folder** in your terminal.
2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install flask openai-whisper torch
    ```
4.  **Run the application:**
    ```bash
    python app.py
    ```
5.  **Open your browser** and navigate to `http://localhost:5000`.

## 📚 References & Documentation

*   **Official GitHub Repository:** [openai/whisper](https://github.com/openai/whisper)
*   **Original Research Paper:** [Robust Speech Recognition via Large-Scale Weak Supervision](https://cdn.openai.com/papers/whisper.pdf) by Alec Radford et al.