"""
Configuration and Persistence Manager.
Loads and saves API keys, audio settings, custom instructions, and UI state to config.json.
"""

import json
import os
from typing import Dict, Any

from humanizer import DEFAULT_INSTRUCTIONS

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

LEGACY_DEFAULT_INSTRUCTIONS = (
    "Label speakers as Speaker 1 and Speaker 2. "
    "Remove false starts and filler words (um, uh, like), "
    "keep technical terms clean, and tag crosstalk as [overlapping]."
)

DEFAULT_CONFIG: Dict[str, Any] = {
    "provider": "deepgram",  # deepgram, gemini, groq, openai, mock
    "deepgram_api_key": "",
    "gemini_api_key": "",
    "groq_api_key": "",
    "openai_api_key": "",
    "deepgram_model": "nova-2",
    "gemini_model": "gemini-flash-lite-latest",
    "groq_model": "llama-3.3-70b-versatile",
    "openai_model": "whisper-1",
    "transcription_language": "hi",
    "selected_audio_device": "",
    "silence_threshold": 0.015,
    "silence_timeout_sec": 1.5,
    "auto_fill_enabled": False,
    "always_on_top": True,
    "custom_instructions": DEFAULT_INSTRUCTIONS,
    "app_mode": "transcriber",  # "transcriber" or "interview_helper"
    "interview_role": "Software Engineer / Technical Candidate",
    "interview_profile_context": (
        "Experienced engineer with strong problem-solving skills, system architecture expertise, "
        "and practical hands-on delivery experience."
    ),
    "interview_style": "Direct Spoken Response (Only the Answer)",
    "interview_auto_answer": True,
    "hotkey_record": "ctrl+alt+r",
    "hotkey_inject": "ctrl+alt+v",
    "hotkey_clear": "ctrl+alt+c",
}


class ConfigManager:
    """Manages application settings persistence."""

    def __init__(self, filepath: str = CONFIG_FILE):
        self.filepath = filepath
        self.config = dict(DEFAULT_CONFIG)
        self.load()

    def load(self):
        """Load settings from JSON file and environment variables."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    self.config.update(loaded)
            except Exception as e:
                print(f"[ConfigManager] Could not read {self.filepath}: {e}")

        if self.config.get("custom_instructions") == LEGACY_DEFAULT_INSTRUCTIONS:
            self.config["custom_instructions"] = DEFAULT_INSTRUCTIONS

        # Seed from environment variables if present
        if not self.config.get("deepgram_api_key") and os.environ.get("DEEPGRAM_API_KEY"):
            self.config["deepgram_api_key"] = os.environ["DEEPGRAM_API_KEY"]
        if not self.config.get("gemini_api_key") and os.environ.get("GEMINI_API_KEY"):
            self.config["gemini_api_key"] = os.environ["GEMINI_API_KEY"]
        if not self.config.get("groq_api_key") and os.environ.get("GROQ_API_KEY"):
            self.config["groq_api_key"] = os.environ["GROQ_API_KEY"]
        if not self.config.get("openai_api_key") and os.environ.get("OPENAI_API_KEY"):
            self.config["openai_api_key"] = os.environ["OPENAI_API_KEY"]

    def save(self):
        """Save settings to config.json."""
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"[ConfigManager] Could not save {self.filepath}: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        self.config[key] = value

    def get_api_key_for_provider(self, provider: str) -> str:
        provider = provider.lower()
        if provider == "deepgram":
            return self.config.get("deepgram_api_key", "")
        elif provider in ("gemini", "google"):
            return self.config.get("gemini_api_key", "")
        elif provider == "groq":
            return self.config.get("groq_api_key", "")
        elif provider == "openai":
            return self.config.get("openai_api_key", "")
        return ""
