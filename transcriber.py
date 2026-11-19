"""
Speech Recognition & Speaker Diarization Engine.
Supports Deepgram Nova-2 (native acoustic diarization), Google Gemini (multimodal audio reasoning),
Groq Whisper, OpenAI Whisper, and an offline test/mock mode.
"""

import os
import requests
from typing import Dict, Any, List, Optional

from humanizer import TRANSCRIPTION_ACCURACY_RULES

SUPPORTED_LANGUAGES = {
    "auto": "Auto-detect",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "hi": "Hindi",
    "ja": "Japanese",
    "zh": "Chinese",
    "pt": "Portuguese",
    "ar": "Arabic",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
}

HINDI_DEVANAGARI_PROMPT = "Transcribe the Hindi audio in Devanagari script (हिन्दी), not Romanized Hindi."


class TranscriptionResult:
    def __init__(self, raw_transcript: str, segments: List[Dict[str, Any]], provider: str):
        self.raw_transcript = raw_transcript
        self.segments = segments  # List of {"speaker": "Speaker 0", "text": "...", "start": 0.0, "end": 1.2}
        self.provider = provider

    def to_speaker_lines(self) -> str:
        """Join adjacent utterances into one unchanged paragraph per speaker turn."""
        paragraphs = []
        for seg in self.segments:
            spk = str(seg.get("speaker") or "?Speaker 1").strip()
            txt = seg.get("text", "").strip()
            if not txt:
                continue
            if paragraphs and paragraphs[-1][0] == spk:
                paragraphs[-1][1] = f"{paragraphs[-1][1]} {txt}"
            else:
                paragraphs.append([spk, txt])

        if not paragraphs and self.raw_transcript:
            return self.raw_transcript
        return "\n".join(f"{speaker}: {text}" for speaker, text in paragraphs)


class SpeechDiarizationEngine:
    """Handles dispatching audio bytes to STT & Diarization backends."""

    def __init__(
        self,
        provider: str = "deepgram",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model_name = model_name

    def set_config(self, provider: str, api_key: Optional[str], model_name: Optional[str] = None):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model_name = model_name

    def transcribe_and_diarize(
        self,
        wav_bytes: bytes,
        custom_instructions: str = "",
        language: str = "auto",
    ) -> TranscriptionResult:
        """Route to appropriate speech recognition and diarization backend."""
        provider = self.provider

        if provider == "deepgram":
            return self._transcribe_deepgram(wav_bytes, language)
        elif provider in ("gemini", "google"):
            return self._transcribe_gemini(wav_bytes, custom_instructions, language)
        elif provider == "groq":
            return self._transcribe_groq(wav_bytes, language)
        elif provider == "openai":
            return self._transcribe_openai(wav_bytes, language)
        elif provider == "mock" or not self.api_key:
            return self._transcribe_mock(wav_bytes)
        else:
            raise ValueError(f"Unsupported transcription provider: {self.provider}")

    def _transcribe_deepgram(self, wav_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        """Deepgram Nova-2 with native multi-speaker acoustic diarization."""
        key = self.api_key or os.environ.get("DEEPGRAM_API_KEY")
        if not key:
            raise ValueError("Deepgram API Key is missing. Please enter it in the Settings panel.")

        url = "https://api.deepgram.com/v1/listen"
        params = {
            "model": self.model_name or "nova-2",
            "diarize": "true",
            "smart_format": "true",
            "punctuate": "true",
            "utterances": "true",
        }
        if language != "auto":
            params["language"] = language
        headers = {
            "Authorization": f"Token {key.strip()}",
            "Content-Type": "audio/wav",
        }

        response = requests.post(url, params=params, headers=headers, data=wav_bytes, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"Deepgram API error ({response.status_code}): {response.text}")

        data = response.json()
        results = data.get("results", {})
        utterances = results.get("utterances", [])

        segments = []
        if utterances:
            for utt in utterances:
                spk_id = utt.get("speaker")
                transcript = utt.get("transcript", "").strip()
                start = utt.get("start", 0.0)
                end = utt.get("end", 0.0)
                if transcript:
                    segments.append({
                        "speaker": f"Speaker {spk_id + 1}" if isinstance(spk_id, int) else "?Speaker 1",
                        "text": transcript,
                        "start": start,
                        "end": end
                    })
        else:
            # Fallback to alternatives
            channels = results.get("channels", [])
            full_text = ""
            if channels:
                alt = channels[0].get("alternatives", [{}])[0]
                full_text = alt.get("transcript", "")
                words = alt.get("words", [])
                curr_spk = None
                curr_words = []
                for w in words:
                    w_spk = w.get("speaker")
                    if curr_spk is None:
                        curr_spk = w_spk
                    if w_spk != curr_spk:
                        speaker = f"Speaker {curr_spk + 1}" if isinstance(curr_spk, int) else "?Speaker 1"
                        segments.append({
                            "speaker": speaker,
                            "text": " ".join(curr_words),
                        })
                        curr_words = []
                        curr_spk = w_spk
                    curr_words.append(w.get("punctuated_word", w.get("word", "")))
                if curr_words:
                    speaker = f"Speaker {curr_spk + 1}" if isinstance(curr_spk, int) else "?Speaker 1"
                    segments.append({
                        "speaker": speaker,
                        "text": " ".join(curr_words),
                    })
            if not segments and full_text:
                segments.append({"speaker": "?Speaker 1", "text": full_text})

        combined_text = "\n".join(f"{s['speaker']}: {s['text']}" for s in segments)
        return TranscriptionResult(raw_transcript=combined_text, segments=segments, provider="Deepgram Nova-2")

    def _transcribe_gemini(
        self,
        wav_bytes: bytes,
        custom_instructions: str = "",
        language: str = "auto",
    ) -> TranscriptionResult:
        """Google Gemini multimodal audio diarization and transcription."""
        key = self.api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("Google Gemini API Key is missing. Please enter it in the Settings panel.")

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key.strip())
        model_id = self.model_name or "gemini-flash-lite-latest"

        system_prompt = (
            "You are a careful audio transcriber and speaker diarization specialist. Listen to the audio itself; "
            "do not infer words from context when they are not audible. Use distinct speaker labels only when "
            "voice evidence supports a change, and mark uncertain speaker labels with '?'.\n\n"
            f"{TRANSCRIPTION_ACCURACY_RULES}\n"
            "Return only speaker-labeled transcript lines, with no preamble, markdown fence, or commentary."
        )

        user_content = "Please transcribe and diarize this system audio recording."
        if language == "hi":
            user_content += f" {HINDI_DEVANAGARI_PROMPT}"
        elif language != "auto":
            language_name = SUPPORTED_LANGUAGES.get(language, language)
            user_content += f" Transcribe in {language_name}."
        if custom_instructions:
            user_content += f"\nRequested transcript style and formatting:\n{custom_instructions}"

        response = client.models.generate_content(
            model=model_id,
            contents=[
                types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                user_content
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.0,
            )
        )

        text = response.text.strip() if response.text else ""
        segments = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                spk, txt = line.split(":", 1)
                segments.append({"speaker": spk.strip(), "text": txt.strip()})
            else:
                segments.append({"speaker": "?Speaker 1", "text": line})

        return TranscriptionResult(raw_transcript=text, segments=segments, provider=f"Google Gemini ({model_id})")

    def _transcribe_groq(self, wav_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        """Transcribe with Groq Whisper without inferring speaker identities."""
        key = self.api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError("Groq API Key is missing. Please enter it in the Settings panel.")

        from groq import Groq
        client = Groq(api_key=key.strip())

        # 1. Transcribe audio
        transcription_options = {
            "file": ("audio.wav", wav_bytes),
            "model": "whisper-large-v3",
            "response_format": "verbose_json",
        }
        if language != "auto":
            transcription_options["language"] = language
        if language == "hi":
            transcription_options["prompt"] = HINDI_DEVANAGARI_PROMPT
        transcription = client.audio.transcriptions.create(
            **transcription_options
        )

        raw_text = getattr(transcription, "text", "") or ""
        segments_raw = getattr(transcription, "segments", []) or []

        segments = []
        for segment in segments_raw:
            if isinstance(segment, dict):
                text = segment.get("text", "")
                start = segment.get("start")
                end = segment.get("end")
            else:
                text = getattr(segment, "text", "")
                start = getattr(segment, "start", None)
                end = getattr(segment, "end", None)
            text = (text or "").strip()
            if text:
                segments.append({
                    "speaker": "?Speaker 1",
                    "text": text,
                    "start": start,
                    "end": end,
                })

        if not segments and raw_text.strip():
            segments.append({"speaker": "?Speaker 1", "text": raw_text.strip()})

        formatted = "\n".join(f"{seg['speaker']}: {seg['text']}" for seg in segments)
        return TranscriptionResult(
            raw_transcript=formatted,
            segments=segments,
            provider="Groq Whisper (speaker identity unavailable)"
        )

    def _transcribe_openai(self, wav_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        """OpenAI Whisper STT."""
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API Key is missing. Please enter it in the Settings panel.")

        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {key.strip()}"}
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {"model": self.model_name or "whisper-1", "response_format": "verbose_json"}
        if language != "auto":
            data["language"] = language
        if language == "hi":
            data["prompt"] = HINDI_DEVANAGARI_PROMPT

        resp = requests.post(url, headers=headers, files=files, data=data, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI error ({resp.status_code}): {resp.text}")

        res_json = resp.json()
        raw_text = res_json.get("text", "")
        segments = [{"speaker": "?Speaker 1", "text": raw_text}] if raw_text else []
        formatted = f"?Speaker 1: {raw_text}" if raw_text else ""
        return TranscriptionResult(raw_transcript=formatted, segments=segments, provider="OpenAI Whisper (no diarization)")

    def _transcribe_mock(self, wav_bytes: bytes) -> TranscriptionResult:
        """
        Demo/Test Mode: Generates a realistic meeting sample based on captured audio signal
        when no API key is configured yet.
        """
        duration = len(wav_bytes) / 32000.0  # Approx seconds for 16kHz 16-bit mono
        sample_dialogues = [
            ("Speaker 1", "We need to push the deployment to Thursday. Are you okay with that, Sarah?"),
            ("Speaker 2", "Yeah, Thursday works for me, but let's check with engineering first."),
            ("Speaker 1", "Agreed. I will sync with Alex on the CI pipeline and confirm by 2 PM."),
            ("Speaker 2", "Sounds good. Make sure we tag crosstalk as [overlapping] if needed.")
        ]
        
        # Pick lines based on duration
        count = min(len(sample_dialogues), max(1, int(duration // 2) + 1))
        selected = sample_dialogues[:count]
        segments = [{"speaker": s, "text": t} for s, t in selected]
        formatted = "\n".join(f"{s['speaker']}: {s['text']}" for s in segments)
        return TranscriptionResult(
            raw_transcript=formatted,
            segments=segments,
            provider="Simulated Test Loopback"
        )
