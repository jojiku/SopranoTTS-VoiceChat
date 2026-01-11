# SopranoTTS Voice Chat

## 🎬 Demo
<table class="center">
  <tr style="font-weight: bolder;text-align:center;">
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/d4c5789e-1d4e-46a6-9968-22582a02a198 controls preload></video>
    </td>
  </tr>
</table>

## ✨ Features

| Feature | Description | Technologies
|---------|-------------|-------------
| ➡️ **Minimal ~300ms latency** | <ul><li>Streaming LLM, STT</li><li>User response prediction</li><li>End-of-turn detection model</li></ul> | SopranoTTS, Faster Whisper, any local or API LLM. Fine-tuned BERT on parlament discussions ([HuggingFace](https://huggingface.co/KoljaB/SentenceFinishedClassification/tree/main))
| 🔄 **Interruption** | Natural system interruption during speech | Internal logic
| 🎯 **Addressee detection** | Understands when you're talking to it vs someone else | Fine-tuned BERT on conversations ([HuggingFace](https://huggingface.co/Silxxor/Lucy-addressee-detector))

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│  Microphone │────▶│  RealtimeSTT    │────▶│    Turn     │
│ (WebSocket) │     │ (Faster Whisper)│     │  Detection  │
└─────────────┘     └─────────────────┘     └──────┬──────┘
                                                   │
                      ┌──────────────┐             │
                      │  Addressee   │◀────────────┤
                      │  Detector    │             │
                      └──────┬───────┘             │
                             │                     │
                             ▼                     ▼
                      ┌──────────────┐     ┌─────────────┐
                      │     LLM      │◀────│  Pipeline   │
                      │  (Streaming) │     │  Manager    │
                      └──────┬───────┘     └─────────────┘
                             │
                             ▼
                      ┌──────────────┐     ┌─────────────┐
                      │  RealtimeTTS │────▶│   Speaker   │
                      │ (SopranoTTS) │     │  (WebSocket)│
                      └──────────────┘     └─────────────┘
```


## 📋 Requirements

- **Python** 3.10
- **CUDA** 12.1 (recommended, ~4GB VRAM)
- **Node.js** 18+ (for frontend)
- **Poetry** 1.8+ for dependency management

## 🚀 Installation

### 1. Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Clone and install

```bash
git clone https://github.com/jojiku/SopranoTTS-VoiceChat.git
cd SopranoTTS-VoiceChat
poetry install
```

### 3. Configure environment

```bash
# In the code folder:
cd code
cp .env.template .env
```

### 4. Install frontend

```bash
# In the code folder:
npm install
```

## ⚙️ Configuration

Edit `.env` as needed:
```env
# Language
APP_LANG=en

# LLM backend (choose one)
LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1
# OLLAMA_BASE_URL=http://127.0.0.1:11434
# OPENAI_API_KEY=sk-... 
# GEMINI_API_KEY=...
# GROQ_API_KEY=gsk_...

# GPU architecture
TORCH_CUDA_ARCH_LIST=7.5
```

## 🎮 Usage

### Run the server
```bash
poetry run python server.py
```

### Open the interface

Navigate to `http://localhost:3000` in your browser.

## 📁 Project Structure

```
SopranoTTS-VoiceChat/
├── server.py                  # FastAPI WebSocket server
├── speech_pipeline_manager.py # LLM + TTS orchestration
├── audio_module.py            # TTS processing (SopranoTTS)
├── audio_in.py                # Input audio processing
├── transcribe.py              # STT processing (Whisper)
├── llm_module.py              # Multi-backend LLM interface
├── addressee_detector.py      # "Is this directed at me or not?"
├── turndetect.py              # End-of-turn prediction
├── soprano_engine.py          # SopranoTTS engine wrapper
├── pyproject.toml             # Poetry configuration
├── static/                    # Frontend sources
├── dist/                      # Built frontend (generated on startup)
└── resources/                 # Prompt storage
```

## 📊 Performance

Metrics with 6 GB VRAM on 1660 TI from user's last word to first system audio chunk:

| Component | Latency | Memory
|-----------|---------|---------
| STT (Whisper base) | ~100ms | ~1000 MB
| LLM (any) | ~150ms TTFT | ~3 GB
| TTS (SopranoTTS) | ~40ms | ~200 MB
| Turn detection (Rubert) | ~20ms | ~100 MB
| Addressee detection (Rubert) | ~20ms | ~100 MB
| **Full pipeline** | **~300ms** | **~5 GB**


## 🙏 Acknowledgments
Awesome code from Kolja Beigel and model from ekwek1
- https://github.com/KoljaB/RealtimeTTS
- https://github.com/KoljaB/RealtimeVoiceChat
- https://github.com/KoljaB/RealtimeSTT
- https://github.com/ekwek1/soprano

## License 📜
**MIT License**
