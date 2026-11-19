import json
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from audio_capture import AudioLoopbackPipeline
from config_manager import ConfigManager, LEGACY_DEFAULT_INSTRUCTIONS
from humanizer import DEFAULT_INSTRUCTIONS, FormattingHumanizer
from transcriber import HINDI_DEVANAGARI_PROMPT, SpeechDiarizationEngine, TranscriptionResult


class HumanizerAccuracyTests(unittest.TestCase):
    def test_local_fallback_preserves_filler_and_spoken_words(self):
        source = "Speaker 1: I, um, wanted to go, but I mean, I stayed."
        output = FormattingHumanizer().format_transcript(source, use_llm=False)

        self.assertEqual(
            output,
            "Speaker 1: I, um, wanted to go, but I mean, I stayed.",
        )

    def test_local_fallback_preserves_recognized_punctuation_and_markers(self):
        source = "Speaker 1: wait--\n[inaudible 00:00:02]"

        self.assertEqual(
            FormattingHumanizer().format_transcript(source, use_llm=False),
            source,
        )


class SpeakerParagraphTests(unittest.TestCase):
    def test_joins_contiguous_same_speaker_and_splits_on_speaker_change(self):
        result = TranscriptionResult(
            raw_transcript="",
            segments=[
                {"speaker": "Speaker 1", "text": "Good morning,"},
                {"speaker": "Speaker 1", "text": "everyone."},
                {"speaker": "Speaker 2", "text": "Good morning!"},
                {"speaker": "Speaker 1", "text": "Let's begin."},
            ],
            provider="test",
        )

        self.assertEqual(
            result.to_speaker_lines(),
            "Speaker 1: Good morning, everyone.\n"
            "Speaker 2: Good morning!\n"
            "Speaker 1: Let's begin.",
        )


class ConfigurationAccuracyTests(unittest.TestCase):
    def test_old_default_rules_are_upgraded_without_overwriting_custom_rules(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = f"{temp_dir}\\config.json"
            with open(config_path, "w", encoding="utf-8") as config_file:
                json.dump({"custom_instructions": LEGACY_DEFAULT_INSTRUCTIONS}, config_file)
            config = ConfigManager(config_path)

        self.assertEqual(config.get("custom_instructions"), DEFAULT_INSTRUCTIONS)

    def test_transcription_language_defaults_to_hindi(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ConfigManager(f"{temp_dir}\\config.json")

        self.assertEqual(config.get("transcription_language"), "hi")


class AudioCaptureAccuracyTests(unittest.TestCase):
    def test_out_of_phase_stereo_does_not_cancel_to_silence(self):
        left = np.array([0.2, -0.2, 0.3, -0.3], dtype=np.float32)
        stereo = np.column_stack((left, -left))

        mono = AudioLoopbackPipeline._downmix_audio(stereo)

        np.testing.assert_array_equal(mono, left)

    def test_short_responses_are_not_discarded_by_default(self):
        pipeline = AudioLoopbackPipeline()

        self.assertEqual(pipeline.min_speech_duration_sec, 0.25)


class ProviderAccuracyTests(unittest.TestCase):
    def test_openai_request_uses_bearer_key_and_selected_model(self):
        response = Mock(status_code=200)
        response.json.return_value = {"text": "Okay."}
        engine = SpeechDiarizationEngine("openai", "test-key", "whisper-1")

        with patch("transcriber.requests.post", return_value=response) as post:
            result = engine._transcribe_openai(b"wav", language="hi")

        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"],
            f"Bearer {'test-key'}",
        )
        self.assertEqual(post.call_args.kwargs["data"]["model"], "whisper-1")
        self.assertEqual(post.call_args.kwargs["data"]["language"], "hi")
        self.assertEqual(post.call_args.kwargs["data"]["prompt"], HINDI_DEVANAGARI_PROMPT)
        self.assertEqual(result.segments[0]["speaker"], "?Speaker 1")

    def test_deepgram_request_includes_selected_language(self):
        response = Mock(status_code=200)
        response.json.return_value = {"results": {}}
        engine = SpeechDiarizationEngine("deepgram", "test-key")

        with patch("transcriber.requests.post", return_value=response) as post:
            engine._transcribe_deepgram(b"wav", language="hi")

        self.assertEqual(post.call_args.kwargs["params"]["language"], "hi")

    def test_groq_does_not_guess_speakers_from_transcript_text(self):
        groq_client = Mock()
        groq_client.audio.transcriptions.create.return_value = types.SimpleNamespace(
            text="Hello there. Yes.",
            segments=[
                types.SimpleNamespace(start=0.1, end=0.7, text="Hello there."),
                types.SimpleNamespace(start=0.8, end=1.0, text="Yes."),
            ],
        )
        groq_module = types.ModuleType("groq")
        groq_module.Groq = Mock(return_value=groq_client)
        engine = SpeechDiarizationEngine("groq", "test-key")

        with patch.dict(sys.modules, {"groq": groq_module}):
            result = engine._transcribe_groq(b"wav", language="fr")

        groq_client.chat.completions.create.assert_not_called()
        self.assertEqual(groq_client.audio.transcriptions.create.call_args.kwargs["language"], "fr")
        self.assertEqual([seg["speaker"] for seg in result.segments], ["?Speaker 1", "?Speaker 1"])
        self.assertEqual([seg["start"] for seg in result.segments], [0.1, 0.8])

    def test_groq_hindi_request_asks_for_devanagari(self):
        groq_client = Mock()
        groq_client.audio.transcriptions.create.return_value = types.SimpleNamespace(text="नमस्ते")
        groq_module = types.ModuleType("groq")
        groq_module.Groq = Mock(return_value=groq_client)
        engine = SpeechDiarizationEngine("groq", "test-key")

        with patch.dict(sys.modules, {"groq": groq_module}):
            engine._transcribe_groq(b"wav", language="hi")

        request = groq_client.audio.transcriptions.create.call_args.kwargs
        self.assertEqual(request["language"], "hi")
        self.assertEqual(request["prompt"], HINDI_DEVANAGARI_PROMPT)

    def test_deepgram_auto_detect_does_not_send_a_language_override(self):
        response = Mock(status_code=200)
        response.json.return_value = {"results": {}}
        engine = SpeechDiarizationEngine("deepgram", "test-key")

        with patch("transcriber.requests.post", return_value=response) as post:
            engine._transcribe_deepgram(b"wav")

        self.assertNotIn("language", post.call_args.kwargs["params"])

    def test_gemini_prompt_requests_the_selected_language(self):
        client = Mock()
        client.models.generate_content.return_value = types.SimpleNamespace(text="Speaker 1: नमस्ते।")
        google_module = types.ModuleType("google")
        google_genai_module = types.ModuleType("google.genai")
        google_types_module = types.ModuleType("google.genai.types")
        google_genai_module.Client = Mock(return_value=client)
        google_module.genai = google_genai_module
        google_types_module.Part = Mock()
        google_types_module.GenerateContentConfig = Mock()
        google_genai_module.types = google_types_module
        engine = SpeechDiarizationEngine("gemini", "test-key")

        with patch.dict(
            sys.modules,
            {
                "google": google_module,
                "google.genai": google_genai_module,
                "google.genai.types": google_types_module,
            },
        ):
            engine._transcribe_gemini(b"wav", language="hi")

        request_content = client.models.generate_content.call_args.kwargs["contents"][1]
        self.assertIn(HINDI_DEVANAGARI_PROMPT, request_content)


if __name__ == "__main__":
    unittest.main()
