"""
Contextual Humanizer & Formatting Engine.
Applies user-defined custom rules, clean verbatim conventions, false start removal,
crosstalk tagging, and speaker alias mappings to diarized transcripts.
"""

import os
from typing import Optional, Dict, Any


DEFAULT_INSTRUCTIONS = (
    "Clean Verbatim. Label speakers consistently and use bold speaker labels followed by a colon. "
    "Remove only clearly non-meaningful fillers, false starts, and stutters; keep false starts that add information. "
    "Keep emphatic repetitions, meaningful expressions, contractions, and the speaker's grammar. "
    "Expand common slang only when unambiguous. Write 'Okay', 'all right', and 'yes' for affirmative yeah/yep/yup/mm-hmm. "
    "Do not paraphrase or add words. Use [crosstalk] only when overlap is evident."
)

TRANSCRIPTION_ACCURACY_RULES = (
    "Transcription fidelity rules:\n"
    "- Never paraphrase, reconstruct, or add speech that is not present in the input.\n"
    "- Preserve word order, meaning, contractions, curse words, and the speaker's grammar; do not silently correct errors.\n"
    "- Keep speaker turns in order. Never infer a speaker identity from wording alone; mark uncertain labels with '?'.\n"
    "- Do not invent timestamps, inaudible/unintelligible markers, sound events, names, or technical spellings.\n"
    "- Follow the requested full- or clean-verbatim style. In clean verbatim, remove only clearly non-meaningful fillers; "
    "preserve meaningful uses, expressions, and emphatic repetitions.\n"
    "- For clean-verbatim spelling, use 'Okay', 'all right', and 'yes' for affirmative yeah/yep/yup/mm-hmm; "
    "expand common slang only when the intended words are unambiguous.\n"
    "- Use bold speaker labels followed by a colon and one space. Keep labels consistent; use role/name only when clear.\n"
    "- Capitalize sentence beginnings and use appropriate end punctuation, except incomplete speech marked with '--'. "
    "Never use exclamation marks or parentheses for editorial notes.\n"
    "- Use '--' for false starts or incomplete thoughts and a single '-' for an interruption where the speaker continues.\n"
    "- Use double quotation marks for direct speech. Keep paragraphs concise (no more than about 500 characters).\n"
    "- Spell out single-digit numbers; use numerals for larger numbers, dates/times, money, measurements, percentages, "
    "and other standard exceptions. Capitalize AM and PM. Preserve names and technical terms rather than guessing.\n"
    "- Use square brackets for concise, lower-case sound notes only when supported by the source. Timestamped markers "
    "must use bold [HH:MM:SS] format, but only when a reliable audio timestamp is already available.\n"
    "- Bold speaker labels and timestamped markers only; do not bold dialogue."
)


class FormattingHumanizer:
    """Enforces styling rules and clean verbatim on diarized text."""

    def __init__(
        self,
        llm_provider: str = "gemini",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.llm_provider = llm_provider.lower()
        self.api_key = api_key
        self.model_name = model_name

    def format_transcript(
        self,
        raw_transcript: str,
        custom_instructions: str = DEFAULT_INSTRUCTIONS,
        use_llm: bool = True
    ) -> str:
        """
        Format the raw diarized transcript according to custom instructions.
        Uses LLM if available; otherwise falls back to deterministic rule cleaner.
        """
        if not raw_transcript or not raw_transcript.strip():
            return ""

        # If LLM enabled and key present, perform LLM contextual pass
        if use_llm:
            try:
                llm_result = self._format_with_llm(raw_transcript, custom_instructions)
                if llm_result and llm_result.strip():
                    return llm_result.strip()
            except Exception as e:
                print(f"[Humanizer] LLM formatting error ({e}), falling back to rule cleaner.")

        # Fallback to local rule-based clean verbatim engine
        return self._format_with_rules(raw_transcript, custom_instructions)

    def _format_with_llm(self, text: str, instructions: str) -> Optional[str]:
        provider = self.llm_provider
        key = self.api_key

        if provider in ("gemini", "google"):
            key = key or os.environ.get("GEMINI_API_KEY")
            if not key:
                return None
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=key.strip())
            model = self.model_name or "gemini-flash-lite-latest"
            prompt = (
                "You are a conservative transcript formatter. The input is already transcribed; do not re-transcribe "
                "or guess what the audio might have said.\n\n"
                f"{TRANSCRIPTION_ACCURACY_RULES}\n\n"
                f"User Instructions:\n{instructions}\n\n"
                f"Raw Diarized Input:\n{text}\n\n"
                "Apply only the explicitly requested formatting and verbatim-style changes. Preserve all other words "
                "and speaker turns. Return only the formatted transcript, without a preamble or code fence."
            )
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0)
            )
            return response.text.strip() if response.text else None

        elif provider == "groq":
            key = key or os.environ.get("GROQ_API_KEY")
            if not key:
                return None
            from groq import Groq
            client = Groq(api_key=key.strip())
            prompt = (
                "You are a conservative transcript formatter. The input is already transcribed; do not re-transcribe "
                "or infer missing words or speakers.\n\n"
                f"{TRANSCRIPTION_ACCURACY_RULES}\n\n"
                f"User Instructions:\n{instructions}\n\n"
                f"Raw Transcript:\n{text}\n\n"
                "Apply only requested formatting and verbatim-style changes. Preserve all other words and turns. "
                "Return only the formatted transcript, without a preamble or code fence."
            )
            chat = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return chat.choices[0].message.content.strip()

        return None

    def _format_with_rules(self, text: str, instructions: str) -> str:
        """Preserve transcript content and punctuation; only discard blank lines."""
        return "\n".join(line.strip() for line in text.splitlines() if line.strip())
