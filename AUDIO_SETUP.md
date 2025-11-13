# Cluely Audio System Configuration Guide

This guide will help you configure your macOS system to capture both **system audio** (meeting participants) and **your microphone input** simultaneously for real-time transcription.

## 📋 Overview

Cluely uses a multi-channel audio setup to capture:
- **System Audio**: Voices from meeting participants (Zoom, Teams, Meet, etc.)
- **Your Voice**: Your microphone input (MacBook mic or Bluetooth earphones)

This is achieved by combining multiple audio sources into an **Aggregate Device** that Cluely can read from.

---

## 🎯 Prerequisites

- macOS (this guide is for macOS only)
- Administrator access (for installing BlackHole)
- Homebrew installed (for easy installation)

---

## 📦 Step 1: Install BlackHole 16ch

BlackHole is a virtual audio driver that captures system audio output.

### Installation

```
brew install blackhole-16ch
```

**What it does:**
- Creates a virtual audio device that can capture system audio
- Acts as a "loopback" - audio going to speakers is also available as input
- Provides 16 input channels for capturing system audio

**After installation:**
- You'll see "BlackHole 16ch" in Audio MIDI Setup
- No restart required

---

## 🎛️ Step 2: Configure Audio MIDI Setup

### 2.1 Create Aggregate Device

The Aggregate Device combines multiple audio inputs into one virtual device.

1. **Open Audio MIDI Setup**
   - Applications → Utilities → Audio MIDI Setup
   - Or press `Cmd + Space` and search "Audio MIDI Setup"

2. **Create Aggregate Device**
   - Click the **"+"** button at the bottom left
   - Select **"Create Aggregate Device"**

3. **Configure Aggregate Device**
   - In the **"Use Audio Device"** table, check the following:
     - ✅ **BlackHole 16ch** (for system audio)
     - ✅ **MacBook Pro Microphone** (for your voice)
     - ✅ **Your Bluetooth Earphone Input** (if using earphones)
   
   - **Set Clock Source**: Select **"BlackHole 16ch"** from the dropdown
   - **Enable Drift Correction**: 
     - ✅ Check for microphone devices (MacBook mic, earphone mic)
     - ❌ Uncheck for BlackHole 16ch (it's the clock source)

4. **Verify Channel Count**
   - The Aggregate Device should show:
     - **18 input channels** (16 from BlackHole + 1 MacBook mic + 1 earphone mic)
     - You can see this in the left sidebar: "18 ins / 18 outs"

### 2.2 Create Multi-Output Device (Optional but Recommended)

This allows you to hear audio while Cluely captures it.

1. **Create Multi-Output Device**
   - Click the **"+"** button
   - Select **"Create Multi-Output Device"**

2. **Configure Multi-Output Device**
   - In the device list, check:
     - ✅ **BlackHole 16ch** (so Cluely can capture it)
     - ✅ **Your Speakers/Earphones** (so you can hear)
   
   - **Set Primary Device**: Select **"BlackHole 16ch"**
   - **Enable Drift Correction**: Check for your speakers/earphones

**Why this is useful:**
- Meeting audio goes to both BlackHole (for Cluely) and your speakers (for you)
- You can hear the meeting while Cluely transcribes it

---

## ⚙️ Step 3: Configure System Settings

### 3.1 Set Audio Output

1. Open **System Settings** → **Sound**
2. Go to **Output** tab
3. Select **"Multi-Output Device"** (or "BlackHole 16ch" if you only want Cluely to capture)

### 3.2 Set Audio Input

1. In **System Settings** → **Sound**
2. Go to **Input** tab
3. Select **"Aggregate Device"**
4. **Test your microphone:**
   - Speak into your mic
   - Watch the "Input level" meter - it should move when you speak
   - If it doesn't move, check microphone permissions (see Troubleshooting)

### 3.3 Configure Meeting App

**For Zoom/Google Meet/Microsoft Teams:**

1. Open your meeting app's audio settings
2. Set **Speaker/Output** to **"Multi-Output Device"** (or "BlackHole 16ch")
3. This routes meeting audio to BlackHole, which feeds the Aggregate Device

---

## 🧪 Step 4: Test Your Setup

Before running Cluely, verify your audio configuration is working.

### Run the Test Script

cd /path/to/free-cluely
source .venv/bin/activate
python test_audio_device.py### What to Look For

**✅ Good Results:**
- Script finds "Aggregate Device" with 18 input channels
- When you **speak**, you see channels **16 and/or 17** showing activity
- When you **play system audio**, you see channels **0-15** showing activity

**❌ Common Issues:**
- **No channels active**: Aggregate Device not configured correctly
- **Only channels 0-15 active**: Mic not included in Aggregate Device
- **Only channels 16-17 active**: System audio not routing to BlackHole
- **Stream open fails**: Permission issue (see Troubleshooting)

### Example Output (Good)
