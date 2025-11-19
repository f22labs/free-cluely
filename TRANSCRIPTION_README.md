# Real-Time Transcription Setup Guide

## Overview

The transcription feature uses **RealtimeSTT** to provide real-time speech-to-text transcription during meetings, interviews, and conversations. The service runs as a Python backend that communicates with the Electron app via JSON over stdin/stdout, enabling low-latency transcription with Whisper-based models.

### Key Features
- **Real-time transcription** with minimal latency
- **Multiple Whisper model support** (small.en, large-v2, etc.)
- **Automatic transcript file generation** with timestamps
- **Microphone testing** to verify audio setup

---

## Quick Installation

### Using Setup Script (Recommended)

The easiest way to set up the transcription service is using the automated setup script:

```bash
# Make the script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

The setup script will:
- Check for all prerequisites (Node.js, Python 3.11+, npm, pip)
- Install Node.js dependencies
- Install Python dependencies (including RealtimeSTT)
- Create `.env` file template
- Make Python scripts executable

---

## Manual Installation

If you prefer to install manually or the setup script encounters issues:

### 1. Prerequisites

- **Python 3.11 or above** (required)
  ```bash
  python3 --version  # Should show 3.11 or higher
  ```

- **Node.js and npm** (for the Electron app)
  ```bash
  node --version
  npm --version
  ```

### 2. Install Python Dependencies

```bash
# Install RealtimeSTT and other dependencies
pip3 install -r requirements.txt
```

**Required Python Package:**
- `realtimestt==0.3.104` (specified in requirements.txt)

### 3. Hugging Face Setup (If Needed)

RealtimeSTT uses Whisper models that are downloaded from Hugging Face. If you encounter authentication errors when downloading models, you may need to log in:

```bash
# Install huggingface-cli if not already installed
pip3 install huggingface-hub

# Login to Hugging Face (creates token for model downloads)
huggingface-cli login
```

**Note:** This is typically only needed if:
- You're using private/gated models
- You're behind a firewall that requires authentication
- You want to use your own Hugging Face account for model caching

For most users, models will download automatically without login.

### 4. Environment Configuration

Create a `.env` file in the project root with your Gemini API key:

```env
# Google Gemini API Key (required for AI features)
GEMINI_API_KEY=your_api_key_here
```

Get your API key from: [Google AI Studio](https://makersuite.google.com/app/apikey)

### 5. Test Microphone

Before using the transcription service, test your microphone setup:

```bash
python3 mic_testing.py
```

This script will:
- List all available audio input/output devices
- Record 10 seconds of audio
- Play back the recording
- Ask for confirmation that audio is working

**Troubleshooting:**
- If you see "PyAudio" errors, install system audio libraries:
  - **macOS:** `brew install portaudio`
  - **Linux:** `sudo apt-get install portaudio19-dev python3-pyaudio`
  - **Windows:** Usually handled automatically by pip

---

## Dependencies Summary

| Component | Version/Requirement | Purpose |
|-----------|---------------------|---------|
| **Python** | 3.11+ | Required for RealtimeSTT service |
| **realtimestt** | 0.3.104 | Real-time speech-to-text library |
| **PyAudio** | Latest | Audio input/output handling |
| **Gemini API Key** | (in .env) | AI-powered transcription analysis |
| **Hugging Face CLI** | (optional) | Model authentication if needed |

---

## Running the Transcription Service

The transcription service is automatically started by the Electron app when you initiate a transcription session. You don't need to run it manually.

However, for debugging purposes, you can test it directly:

```bash
# Test the service with debug output
bash debug_transcription.sh

# Or run directly
python3 realtime_stt_service.py
```

---

## How It Works

1. **User initiates transcription** from the Electron app UI
2. **Electron spawns Python process** running `realtime_stt_service.py`
3. **RealtimeSTT service**:
   - Initializes Whisper model (downloads from Hugging Face if needed)
   - Opens microphone stream
   - Processes audio in real-time
   - Sends transcription updates via JSON to Electron
4. **Electron app** receives and displays transcriptions
5. **Transcript files** are saved to `transcripts/` directory with timestamps

---

## Troubleshooting

### SSL Certificate Verification Error (macOS)

**Error:** `[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate`

This error occurs when RealtimeSTT tries to download the Silero VAD model from GitHub but Python's SSL certificates aren't properly configured on macOS.

**Fix:**
```bash
# Run the Python certificate installer (adjust version if needed)
/Applications/Python\ 3.11/Install\ Certificates.command
```

This installs the proper root certificates for Python's SSL module. After running this, restart your transcription service.

**Alternative Fix (if above doesn't work):**
```bash
# Install certificates using pip
pip3 install --upgrade certifi
```

### "RealtimeSTT not installed" Error
```bash
pip3 install realtimestt==0.3.104
```

### "PyAudio" Installation Errors
```bash
# macOS
brew install portaudio
pip3 install --force-reinstall pyaudio

# Linux
sudo apt-get install portaudio19-dev
pip3 install --force-reinstall pyaudio
```

### "Model download failed" Error
```bash
# Try logging into Hugging Face
huggingface-cli login
```

### "Silero VAD model not found in cache" Error

If you see: `It looks like there is no internet connection and the repo could not be found in the cache`

**Solutions:**
1. **Fix SSL certificates** (see SSL Certificate Verification Error above)
2. **Manually download the model:**
   ```bash
   python3 -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=True)"
   ```
3. **Check internet connection** and firewall settings

### Microphone Not Detected
- Run `python3 mic_testing.py` to see available devices
- Check system audio permissions
- Ensure microphone is not being used by another application

### Low Audio Levels
- Check microphone volume in system settings
- Move closer to microphone
- Check for background noise interference

---

## Additional Resources

- [RealtimeSTT Documentation](https://github.com/KoljaB/RealtimeSTT)
- [Whisper Models on Hugging Face](https://huggingface.co/models?library=transformers&sort=downloads&search=whisper)
- [Google Gemini API Documentation](https://ai.google.dev/docs)
