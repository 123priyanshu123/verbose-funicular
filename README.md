# 🎧 Internal Audio Loopback Transcriber & Diarization Agent

A real-time AI desktop agent engineered to capture **internal digital audio directly from your system output** (bypassing room microphone noise completely). It transcribes, diarizes distinct speakers, cleans and humanizes dialogue according to your custom rules, and automatically injects the formatted dialogue straight into your active focused window or form field via simulated keystrokes.

---

## 🏗️ Architecture Overview

```
                          ┌──────────────────────────┐
                          │    System Audio Output   │
                          │ (Earbuds / Speakers / OS) │
                          └────────────┬─────────────┘
                                       │
                         [ WASAPI / System Loopback ]
                                       │
                                       ▼
┌──────────────────┐      ┌──────────────────────────┐
│  Background GUI  │ ───► │  Audio Buffer Manager    │
│ (Instructions /  │      │  (VAD & Chunk Splitter)  │
│    API Key)      │      └────────────┬─────────────┘
└──────────────────┘                   │
                                       ▼
                          ┌──────────────────────────┐
                          │   Speech Recognition &   │
                          │   Speaker Diarization    │
                          └────────────┬─────────────┘
                                       │
                                       ▼
                          ┌──────────────────────────┐
                          │ LLM Formatting/Humanizer │
                          │ (Rules & Style Engine)   │
                          └────────────┬─────────────┘
                                       │
                         [ Global Hotkey / Auto-Fill ]
                                       │
                                       ▼
                          ┌──────────────────────────┐
                          │ Active Text Box (Focused)│
                          └──────────────────────────┘
```

The application is structured into four core layers:

1. **Layer 1: Internal Audio Capture (WASAPI Loopback)**
   - Uses native Windows WASAPI loopback streams via `soundcard`.
   - Captures crisp, raw digital audio streams sent to your playback device (Bluetooth earbuds, USB headsets, or speakers).
   - Zero physical room noise or mic interference.
   - Built-in Voice Activity Detection (VAD) with energy thresholding (RMS) and silence detection.

2. **Layer 2: Multi-Speaker Diarization & Speech-to-Text**
   - **Deepgram Nova-2** (Native acoustic diarization with speaker clustering and timestamps).
   - **Google Gemini (gemini-2.5-flash / gemini-2.0-flash)** (Native multimodal audio reasoning with speaker tags).
   - **Groq Whisper** (`whisper-large-v3`; speaker labels are marked uncertain because this provider does not diarize).
   - **OpenAI Whisper** (`whisper-1`; speaker labels are marked uncertain because this provider does not diarize).
   - **Simulated Test Mode** (Offline pipeline test without needing an API key immediately).

3. **Layer 3: Contextual Humanizer & Formatting Engine**
   - Joins consecutive utterances from the same detected speaker into one paragraph and starts a new paragraph when the speaker changes.
   - Preserves the speech recognizer's punctuation and transcript markers; the display formatter does not rewrite, correct, or add markup.

4. **Layer 4: Desktop Overlay & Active Window Injector**
   - Modern dark-mode floating UI (`CustomTkinter`) with **Pin on Top** mode.
   - Live VU Audio Level Meter monitoring internal sound volume.
   - Global OS Hotkeys (`Ctrl+Alt+R`, `Ctrl+Alt+V`, `Ctrl+Alt+C`).
   - **Hands-Free Auto-Fill**: Automatically injects finalized transcript into whichever window, test field, chat box, or document is currently active on your screen.

---

## 🎛️ Dual Modes: Transcriber vs. AI Interview Helper

The application features a top-level mode selector enabling you to switch instantly between two core modes:

### 1. 🎙️ Audio Transcriber Mode
- **Purpose**: Real-time internal audio transcription and multi-speaker diarization for meetings, video calls, podcasts, and recordings.
- **Features**:
  - Automatically tags distinct speakers (`Speaker 1`, `Speaker 2`).
  - Supports Clean Verbatim, Full Verbatim, or custom formatting guidelines.
  - Multi-language support (Hindi in Devanagari script, English, Spanish, French, German, etc.).
  - Auto-fill or manual injection (`Ctrl+Alt+V`) into active forms or documents.

### 2. 💼 AI Interview Helper Mode (Chat Stream)
- **Purpose**: Real-time conversational interview copilot that captures the interviewer's voice directly from your system loopback (earbuds/headphones) and provides immediate candidate responses in a sequential chat feed.
- **Features**:
  - **Conversational Chat Stream**:
    - **Interviewer Message**: Displays each question asked by the interviewer with timestamps and question tags (`👤 Interviewer [16:05:12] • Q#1`).
    - **Candidate Response Message**: Renders the AI response directly below it in the next chat bubble.
  - **Direct Response (Only the Answer)**:
    - Provides **strictly the articulate first-person spoken answer** without meta-headers, bullet categories, or boilerplate filler. Ready to be spoken immediately or injected into a chat.
  - **Interactive Chat Input Bar**:
    - Type or paste questions directly (e.g. from Zoom chat, Teams, or coding tests) and press `Enter` to generate instant answers.
  - **Bubble Actions**:
    - Each AI response bubble includes instant **`📋 Copy`** and **`⚡ Inject`** (`Ctrl+Alt+V`) buttons.
  - **Delivery Styles**:
    - *Direct Spoken Response (Only the Answer)* (Default)
    - *STAR Method (Situation, Task, Action, Result)*
    - *Technical Deep-Dive / Code & Architecture*
    - *Executive / Concise Summary*
  - **Candidate Profile & Context**:
    - Save your Target Role (e.g. *Senior Backend Engineer*) and Resume Highlights so generated answers seamlessly reflect your experience and tech stack.
  - **⚡ Auto-Answer & Instant Injection**:
    - Generates responses automatically as the interviewer finishes speaking, with hands-free auto-fill if enabled.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **OS**: Windows 10 or 11
- **Python**: Python 3.10+ (Python 3.12 verified)

### 2. Installation
Clone or navigate to the workspace directory:
```powershell
cd "c:\Users\priya\OneDrive\Desktop\VSCODE\AI Audio transcrtipter"
```

Install dependencies:
```powershell
pip install -r requirements.txt
```

### 3. Verify Audio Loopback on Your Machine
Inspect your system's detected loopback audio devices:
```powershell
python main.py --list-devices
```

Run a 2-second capture test to confirm the digital stream is detected:
```powershell
python main.py --test
```

### 4. Launch the Desktop Application
```powershell
python main.py
```

---

## ⚙️ Configuration & API Keys

Open the **⚙️ Providers & Settings** tab in the desktop application:
1. Select your preferred transcription engine:
   - **Deepgram Nova-2 (Recommended)**: Industry standard for real-time acoustic speaker diarization. Enter your `Deepgram API Key`.
   - **Google Gemini (gemini-2.5-flash)**: State-of-the-art multimodal audio understanding and speaker diarization. Enter your `Gemini API Key`.
   - **Groq Whisper**: Ultra-fast Whisper large-v3 inference. Enter your `Groq API Key`.
   - **OpenAI Whisper**: Standard Whisper STT. Enter your `OpenAI API Key`.
   - **Simulated Test Mode**: Runs offline without an API key to test hotkeys, loopback, and auto-fill injection.
2. Select a transcription language (Hindi is selected by default). For Hindi audio, the app requests output in Devanagari script rather than Romanized Hindi. You can also choose **Auto-detect**, English, Spanish, French, German, Japanese, Chinese, Portuguese, Arabic, Italian, Korean, or Russian.
3. Click **💾 Save Settings & API Keys** to save the provider, language, and keys in `config.json`. Language availability may vary by provider and model.

---

## ⌨️ Global Hotkeys & Controls

| Shortcut | Action | Description |
|---|---|---|
| `Ctrl + Alt + R` | **Toggle Recording** | Starts or stops internal audio loopback capture. |
| `Ctrl + Alt + V` | **Inject to Active Window** | Emulates `Ctrl+V` to paste the last formatted transcript into your focused window. |
| `Ctrl + Alt + C` | **Clear Transcript** | Clears the live transcript feed. |
| **⚡ Auto-Fill Switch** | **Hands-Free Injection** | When toggled ON, the agent automatically pastes results into your active window upon speech completion without pressing any keys. |
| **📌 Pin Top** | **Overlay Float** | Keeps the agent floating above Zoom, Teams, Chrome, or code editors. |

---

## 📝 Custom Rules & Formatting Examples

In the **📝 Custom Rules & Formatting** tab, you can customize the formatting engine with any prompt or assessment guideline:

### Default Clean Verbatim:
```text
Clean Verbatim. Remove only clearly non-meaningful fillers, false starts, and stutters. Keep emphatic repetitions, meaningful expressions, contractions, slang, and the speaker's grammar. Do not paraphrase.
```

### Full Verbatim:
```text
Transcribe every audible word exactly as spoken, including fillers, false starts, stutters, repetitions, slang, contractions, and speech errors. Do not correct grammar or paraphrase.
```

### Interview Mode:
```text
Label speakers as Interviewer and Interviewee when their roles are clear. Remove only clearly non-meaningful fillers, keep the speaker's grammar and wording, and do not paraphrase.
```

Speech recognition quality depends on the selected provider, audio clarity, and speaker overlap. Groq Whisper and OpenAI Whisper do not provide speaker diarization here; the app marks speaker labels as uncertain rather than guessing from dialogue. When no formatting LLM is available, the local formatter preserves transcript wording instead of deleting words based on fixed filler-word heuristics.

---

## 🎧 Listening Through Bluetooth Earbuds

When you connect Bluetooth earbuds (e.g. AirPods, Sony WH-1000XM, Galaxy Buds):
1. Windows routes system audio to your earbuds.
2. In the desktop app, leave the device selector on **Default System Playback Loopback** (or select your specific earbuds from the dropdown).
3. The agent taps directly into the audio buffer being transmitted over Bluetooth to your ears.
4. Meeting participants on Zoom, Meet, or Teams will be transcribed crisply—even if your physical room is noisy or your microphone is muted.
