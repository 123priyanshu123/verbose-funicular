"""
Desktop Overlay Application & GUI for Internal Audio Loopback Transcription & Diarization Agent.
Built with CustomTkinter for a sleek, modern, dark-mode floating interface.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk

from audio_capture import AudioLoopbackPipeline, get_loopback_devices, get_default_loopback_device
from transcriber import SUPPORTED_LANGUAGES, SpeechDiarizationEngine, TranscriptionResult
from humanizer import FormattingHumanizer, DEFAULT_INSTRUCTIONS
from injector import ActiveWindowInjector, HotkeyManager
from config_manager import ConfigManager, CONFIG_FILE
from interview_helper import (
    InterviewHelperEngine,
    ANSWER_STYLES,
    DEFAULT_ROLE,
    DEFAULT_PROFILE_CONTEXT,
)


ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class AudioAgentApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Audio Loopback Transcriber & Diarization Agent")
        self.geometry("960x780")
        self.minsize(760, 620)

        # Config
        self.config_mgr = ConfigManager()
        self.attributes("-topmost", self.config_mgr.get("always_on_top", True))

        # Core Engines
        current_provider = self.config_mgr.get("provider", "deepgram")
        api_key = self.config_mgr.get_api_key_for_provider(current_provider)
        
        self.transcriber = SpeechDiarizationEngine(
            provider=current_provider,
            api_key=api_key,
            model_name=self.config_mgr.get(f"{current_provider}_model", None)
        )
        self.humanizer = FormattingHumanizer(
            llm_provider="gemini" if self.config_mgr.get("gemini_api_key") else current_provider,
            api_key=self.config_mgr.get("gemini_api_key") or api_key
        )
        self.interview_helper = InterviewHelperEngine(
            provider="gemini" if self.config_mgr.get("gemini_api_key") else current_provider,
            api_key=self.config_mgr.get("gemini_api_key") or api_key,
            model_name=self.config_mgr.get("gemini_model", None)
        )
        self.pipeline = AudioLoopbackPipeline(
            sample_rate=16000,
            silence_threshold=float(self.config_mgr.get("silence_threshold", 0.015)),
            silence_timeout_sec=float(self.config_mgr.get("silence_timeout_sec", 1.5)),
            on_speech_chunk=self._on_speech_chunk_received,
            on_audio_level=self._on_audio_level_update,
            on_status_change=self._on_status_change_update
        )

        # State
        self.last_formatted_text = ""
        self.last_interview_question = ""
        self.last_interview_answer = ""
        self.interview_question_count = 0
        self.chunk_count = 0
        self.hotkey_mgr = HotkeyManager(
            on_toggle_recording=self.toggle_recording,
            on_inject_last=self.inject_active_window,
            on_clear_transcript=self.clear_transcript
        )

        # Build UI
        self._build_ui()
        self._populate_devices()

        # Start Hotkeys
        self.hotkey_mgr.start()

        # Graceful cleanup on close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        # Master Grid
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # ---------------- 1. Top Header Bar ----------------
        header_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#1a1c23")
        header_frame.grid(row=0, column=0, padx=14, pady=(10, 5), sticky="ew")
        header_frame.grid_columnconfigure(2, weight=1)

        title_label = ctk.CTkLabel(
            header_frame,
            text="🎧 Audio Loopback Diarizer",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#60a5fa"
        )
        title_label.grid(row=0, column=0, padx=(16, 8), pady=10, sticky="w")

        # Mode Selector Segmented Button (Transcriber vs AI Interview Helper)
        current_mode = self.config_mgr.get("app_mode", "transcriber")
        initial_mode_label = "💼 AI Interview Helper" if current_mode == "interview_helper" else "🎙️ Transcriber"
        self.mode_selector = ctk.CTkSegmentedButton(
            header_frame,
            values=["🎙️ Transcriber", "💼 AI Interview Helper"],
            command=self._on_mode_changed,
            font=ctk.CTkFont(size=12, weight="bold"),
            selected_color="#2563eb",
            selected_hover_color="#1d4ed8",
            unselected_color="#374151"
        )
        self.mode_selector.set(initial_mode_label)
        self.mode_selector.grid(row=0, column=1, padx=8, pady=10, sticky="w")

        # Status Badge Pill
        self.status_badge = ctk.CTkLabel(
            header_frame,
            text="● IDLE",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#374151",
            text_color="#9ca3af",
            corner_radius=12,
            padx=12,
            pady=4
        )
        self.status_badge.grid(row=0, column=2, padx=10, pady=10, sticky="w")

        # Top Right Controls: Auto-Fill & Always on Top
        right_header = ctk.CTkFrame(header_frame, fg_color="transparent")
        right_header.grid(row=0, column=2, padx=16, pady=12, sticky="e")

        self.auto_fill_var = ctk.BooleanVar(value=self.config_mgr.get("auto_fill_enabled", False))
        self.auto_fill_switch = ctk.CTkSwitch(
            right_header,
            text="⚡ Auto-Fill Active Window",
            variable=self.auto_fill_var,
            command=self._on_auto_fill_toggled,
            font=ctk.CTkFont(size=12, weight="bold"),
            progress_color="#10b981"
        )
        self.auto_fill_switch.pack(side="left", padx=10)

        self.topmost_var = ctk.BooleanVar(value=self.config_mgr.get("always_on_top", True))
        self.topmost_switch = ctk.CTkSwitch(
            right_header,
            text="📌 Pin Top",
            variable=self.topmost_var,
            command=self._on_topmost_toggled,
            font=ctk.CTkFont(size=12)
        )
        self.topmost_switch.pack(side="left", padx=6)

        # ---------------- 2. Audio Device & VU Meter Bar ----------------
        device_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#1f2937")
        device_frame.grid(row=1, column=0, padx=14, pady=5, sticky="ew")
        device_frame.grid_columnconfigure(1, weight=1)

        dev_label = ctk.CTkLabel(device_frame, text="System Loopback Device:", font=ctk.CTkFont(size=12, weight="bold"))
        dev_label.grid(row=0, column=0, padx=(14, 8), pady=8, sticky="w")

        self.device_combo = ctk.CTkComboBox(
            device_frame,
            values=["Default System Playback Loopback"],
            command=self._on_device_selected,
            width=340
        )
        self.device_combo.grid(row=0, column=1, padx=6, pady=8, sticky="ew")

        refresh_btn = ctk.CTkButton(
            device_frame,
            text="🔄 Refresh",
            width=80,
            command=self._populate_devices,
            fg_color="#374151",
            hover_color="#4b5563"
        )
        refresh_btn.grid(row=0, column=2, padx=6, pady=8)

        test_btn = ctk.CTkButton(
            device_frame,
            text="🔊 Test Loopback",
            width=110,
            command=self._test_loopback_capture,
            fg_color="#2563eb",
            hover_color="#1d4ed8"
        )
        test_btn.grid(row=0, column=3, padx=(6, 14), pady=8)

        # VU Level Meter
        meter_row = ctk.CTkFrame(device_frame, fg_color="transparent")
        meter_row.grid(row=1, column=0, columnspan=4, padx=14, pady=(0, 8), sticky="ew")
        meter_row.grid_columnconfigure(1, weight=1)

        meter_lbl = ctk.CTkLabel(meter_row, text="Internal Sound Level:", font=ctk.CTkFont(size=11), text_color="#9ca3af")
        meter_lbl.grid(row=0, column=0, padx=(0, 8), sticky="w")

        self.vu_bar = ctk.CTkProgressBar(meter_row, height=10, progress_color="#10b981")
        self.vu_bar.grid(row=0, column=1, sticky="ew")
        self.vu_bar.set(0.0)

        # ---------------- 3. Main Center Tabs ----------------
        self.tabview = ctk.CTkTabview(self, corner_radius=10, command=self._on_tab_changed)
        self.tabview.grid(row=2, column=0, padx=14, pady=5, sticky="nsew")

        self.tab_transcribe = self.tabview.add("🎙️ Live Diarization")
        self.tab_interview = self.tabview.add("💼 AI Interview Helper")
        self.tab_instructions = self.tabview.add("📝 Custom Rules & Formatting")
        self.tab_settings = self.tabview.add("⚙️ Providers & Settings")

        self._build_transcription_tab()
        self._build_interview_tab()
        self._build_instructions_tab()
        self._build_settings_tab()

        # Set initial tab according to saved app mode
        if current_mode == "interview_helper":
            self.tabview.set("💼 AI Interview Helper")
        else:
            self.tabview.set("🎙️ Live Diarization")

        # ---------------- 4. Bottom Footer / Hotkey Hint Bar ----------------
        footer_frame = ctk.CTkFrame(self, corner_radius=6, fg_color="#111827", height=32)
        footer_frame.grid(row=3, column=0, padx=14, pady=(4, 10), sticky="ew")
        footer_frame.grid_columnconfigure(1, weight=1)

        self.footer_status = ctk.CTkLabel(
            footer_frame,
            text="Ready. Press Start or hit [Ctrl+Alt+R] to capture system voices.",
            font=ctk.CTkFont(size=11),
            text_color="#9ca3af"
        )
        self.footer_status.grid(row=0, column=0, padx=12, pady=4, sticky="w")

        hotkey_hint = ctk.CTkLabel(
            footer_frame,
            text="Hotkeys: [Ctrl+Alt+R] Record  |  [Ctrl+Alt+V] Auto-Fill  |  [Ctrl+Alt+C] Clear",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#60a5fa"
        )
        hotkey_hint.grid(row=0, column=1, padx=12, pady=4, sticky="e")

    # ---------------- Tab 1: Transcription UI ----------------
    def _build_transcription_tab(self):
        tab = self.tab_transcribe
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # Controls Row
        ctrl_frame = ctk.CTkFrame(tab, fg_color="transparent")
        ctrl_frame.grid(row=0, column=0, pady=(0, 8), sticky="ew")
        ctrl_frame.grid_columnconfigure(0, weight=1)

        self.btn_record = ctk.CTkButton(
            ctrl_frame,
            text="▶ Start Listening (Ctrl+Alt+R)",
            command=self.toggle_recording,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#059669",
            hover_color="#047857",
            height=38
        )
        self.btn_record.grid(row=0, column=0, padx=(0, 8), sticky="ew")

        self.btn_inject = ctk.CTkButton(
            ctrl_frame,
            text="⚡ Inject to Focused Window (Ctrl+Alt+V)",
            command=self.inject_active_window,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#7c3aed",
            hover_color="#6d28d9",
            height=38,
            width=260
        )
        self.btn_inject.grid(row=0, column=1, padx=4)

        btn_copy = ctk.CTkButton(
            ctrl_frame,
            text="📋 Copy",
            command=self.copy_transcript,
            font=ctk.CTkFont(size=12),
            fg_color="#374151",
            hover_color="#4b5563",
            height=38,
            width=80
        )
        btn_copy.grid(row=0, column=2, padx=4)

        btn_clear = ctk.CTkButton(
            ctrl_frame,
            text="🗑️ Clear",
            command=self.clear_transcript,
            font=ctk.CTkFont(size=12),
            fg_color="#dc2626",
            hover_color="#b91c1c",
            height=38,
            width=80
        )
        btn_clear.grid(row=0, column=3, padx=(4, 0))

        # Transcript Text Area
        self.transcript_box = ctk.CTkTextbox(
            tab,
            font=ctk.CTkFont(family="Consolas", size=13),
            wrap="word",
            corner_radius=8
        )
        self.transcript_box.grid(row=1, column=0, sticky="nsew")

        # Initial placeholder instructions
        initial_sample = (
            "--- System Audio Diarizer Ready ---\n"
            "1. Put on your Bluetooth earbuds or play any internal video/call.\n"
            "2. Click 'Start Listening' or hit Ctrl+Alt+R.\n"
            "3. Internal voices will be captured, diarized by speaker, humanized, and displayed here.\n"
            "4. Turn on 'Auto-Fill Active Window' to paste results hands-free into your active form or document!\n"
        )
        self.transcript_box.insert("0.0", initial_sample)

    # ---------------- Tab 2: AI Interview Helper UI (Chat Stream) ----------------
    def _build_interview_tab(self):
        tab = self.tab_interview
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # 1. Top Controls Bar inside Interview Tab
        top_ctrl = ctk.CTkFrame(tab, fg_color="#1e222d", corner_radius=8)
        top_ctrl.grid(row=0, column=0, pady=(0, 6), sticky="ew")
        top_ctrl.grid_columnconfigure(2, weight=1)

        self.btn_interview_record = ctk.CTkButton(
            top_ctrl,
            text="▶ Start Interview Capture (Ctrl+Alt+R)",
            command=self.toggle_recording,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#059669",
            hover_color="#047857",
            height=34,
            width=220
        )
        self.btn_interview_record.grid(row=0, column=0, padx=8, pady=6, sticky="w")

        # Answer Style Selector
        ctk.CTkLabel(top_ctrl, text="Style:", font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=1, padx=(6, 2), pady=6, sticky="w")
        self.combo_interview_style = ctk.CTkComboBox(
            top_ctrl,
            values=[
                "Direct Spoken Response (Only the Answer)",
                "STAR Method (Situation, Task, Action, Result)",
                "Technical Deep-Dive / Code & Architecture",
                "Executive / Concise Summary"
            ],
            command=self._on_interview_style_changed,
            width=260,
            font=ctk.CTkFont(size=11)
        )
        self.combo_interview_style.set(self.config_mgr.get("interview_style", "Direct Spoken Response (Only the Answer)"))
        self.combo_interview_style.grid(row=0, column=2, padx=4, pady=6, sticky="w")

        # Auto-Answer Switch
        self.auto_answer_var = ctk.BooleanVar(value=self.config_mgr.get("interview_auto_answer", True))
        self.auto_answer_switch = ctk.CTkSwitch(
            top_ctrl,
            text="⚡ Auto-Answer",
            variable=self.auto_answer_var,
            command=self._on_auto_answer_toggled,
            font=ctk.CTkFont(size=12, weight="bold"),
            progress_color="#10b981"
        )
        self.auto_answer_switch.grid(row=0, column=3, padx=8, pady=6)

        # Action Buttons
        btn_copy_ans = ctk.CTkButton(
            top_ctrl,
            text="📋 Copy Last",
            command=self.copy_interview_answer,
            font=ctk.CTkFont(size=11),
            fg_color="#374151",
            hover_color="#4b5563",
            height=30,
            width=80
        )
        btn_copy_ans.grid(row=0, column=4, padx=4, pady=6)

        btn_regen = ctk.CTkButton(
            top_ctrl,
            text="🔄 Regenerate",
            command=self.regenerate_interview_answer,
            font=ctk.CTkFont(size=11),
            fg_color="#4b5563",
            hover_color="#6b7280",
            height=30,
            width=85
        )
        btn_regen.grid(row=0, column=5, padx=4, pady=6)

        btn_clear_interview = ctk.CTkButton(
            top_ctrl,
            text="🗑️ Clear Chat",
            command=self.clear_interview,
            font=ctk.CTkFont(size=11),
            fg_color="#dc2626",
            hover_color="#b91c1c",
            height=30,
            width=80
        )
        btn_clear_interview.grid(row=0, column=6, padx=(4, 8), pady=6)

        # 2. Main Center: Chat Stream Feed (Scrollable Frame)
        self.interview_chat_scroll = ctk.CTkScrollableFrame(
            tab,
            fg_color="#10141d",
            corner_radius=10,
            scrollbar_button_color="#374151",
            scrollbar_button_hover_color="#4b5563"
        )
        self.interview_chat_scroll.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        self.interview_chat_scroll.grid_columnconfigure(0, weight=1)

        self.thinking_card = None

        # Add initial welcome notice in chat
        self._add_system_notice(
            "💬 AI Interview Chat Stream Active\n\n"
            "• When your interviewer speaks through your earbuds or meeting audio, their question appears here.\n"
            "• The AI immediately responds in the very next chat message with ONLY the candidate's spoken response.\n"
            "• You can also type or paste questions directly into the input bar below and press Enter!"
        )

        # 3. Bottom Chat Input Bar
        chat_input_frame = ctk.CTkFrame(tab, fg_color="#181d28", corner_radius=8)
        chat_input_frame.grid(row=2, column=0, pady=(6, 4), sticky="ew")
        chat_input_frame.grid_columnconfigure(0, weight=1)

        self.chat_input_entry = ctk.CTkEntry(
            chat_input_frame,
            placeholder_text="Type or paste interviewer question here... (Press Enter to Send)",
            font=ctk.CTkFont(size=12),
            height=34
        )
        self.chat_input_entry.grid(row=0, column=0, padx=(8, 4), pady=6, sticky="ew")
        self.chat_input_entry.bind("<Return>", lambda e: self._on_chat_input_submit())

        btn_send = ctk.CTkButton(
            chat_input_frame,
            text="Send ↵",
            command=self._on_chat_input_submit,
            width=80,
            height=34,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#2563eb",
            hover_color="#1d4ed8"
        )
        btn_send.grid(row=0, column=1, padx=(4, 8), pady=6)

        # 4. Profile & Target Role Configuration Strip
        profile_bar = ctk.CTkFrame(tab, fg_color="#181d28", corner_radius=8)
        profile_bar.grid(row=3, column=0, pady=(0, 2), sticky="ew")
        profile_bar.grid_columnconfigure(1, weight=3)
        profile_bar.grid_columnconfigure(3, weight=5)

        ctk.CTkLabel(profile_bar, text="Target Role:", font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=0, padx=(8, 4), pady=5, sticky="w")
        self.entry_interview_role = ctk.CTkEntry(profile_bar, font=ctk.CTkFont(size=11), width=180)
        self.entry_interview_role.grid(row=0, column=1, padx=4, pady=5, sticky="ew")
        self.entry_interview_role.insert(0, self.config_mgr.get("interview_role", DEFAULT_ROLE))

        ctk.CTkLabel(profile_bar, text="Your Background:", font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=2, padx=(10, 4), pady=5, sticky="w")
        self.entry_interview_context = ctk.CTkEntry(profile_bar, font=ctk.CTkFont(size=11), placeholder_text="e.g. 5 yrs in Python, AWS, Microservices, System Architecture...")
        self.entry_interview_context.grid(row=0, column=3, padx=4, pady=5, sticky="ew")
        self.entry_interview_context.insert(0, self.config_mgr.get("interview_profile_context", DEFAULT_PROFILE_CONTEXT))

        btn_save_profile = ctk.CTkButton(
            profile_bar,
            text="💾 Save Profile",
            command=self._save_interview_profile,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#2563eb",
            hover_color="#1d4ed8",
            height=26,
            width=90
        )
        btn_save_profile.grid(row=0, column=4, padx=(4, 8), pady=5)

    # ---------------- Tab 3: Custom Instructions UI ----------------
    def _build_instructions_tab(self):
        tab = self.tab_instructions
        tab.grid_rowconfigure(2, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        lbl = ctk.CTkLabel(
            tab,
            text="Rule & Style Specification Engine:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        lbl.grid(row=0, column=0, padx=6, pady=(4, 2), sticky="w")

        hint = ctk.CTkLabel(
            tab,
            text="Define how the AI cleans, labels, and formats the dialogue (e.g. Clean Verbatim, custom aliases, timestamp placement, crosstalk).",
            font=ctk.CTkFont(size=12),
            text_color="#9ca3af"
        )
        hint.grid(row=1, column=0, padx=6, pady=(0, 6), sticky="w")

        self.instructions_box = ctk.CTkTextbox(
            tab,
            font=ctk.CTkFont(size=13),
            wrap="word",
            corner_radius=8
        )
        self.instructions_box.grid(row=2, column=0, padx=4, pady=4, sticky="nsew")
        self.instructions_box.insert("0.0", self.config_mgr.get("custom_instructions", DEFAULT_INSTRUCTIONS))

        # Preset Buttons Row
        preset_frame = ctk.CTkFrame(tab, fg_color="transparent")
        preset_frame.grid(row=3, column=0, pady=8, sticky="ew")

        ctk.CTkLabel(preset_frame, text="Presets:", font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=6)

        ctk.CTkButton(
            preset_frame,
            text="Clean Verbatim (Default)",
            command=lambda: self._set_instruction_preset("clean"),
            fg_color="#374151",
            hover_color="#4b5563",
            height=28
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            preset_frame,
            text="Strict Verbatim",
            command=lambda: self._set_instruction_preset("strict"),
            fg_color="#374151",
            hover_color="#4b5563",
            height=28
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            preset_frame,
            text="Interview / Q&A",
            command=lambda: self._set_instruction_preset("interview"),
            fg_color="#374151",
            hover_color="#4b5563",
            height=28
        ).pack(side="left", padx=4)

        btn_save_instructions = ctk.CTkButton(
            preset_frame,
            text="💾 Save Rules",
            command=self._save_instructions,
            fg_color="#2563eb",
            hover_color="#1d4ed8",
            height=28,
            width=100
        )
        btn_save_instructions.pack(side="right", padx=6)

    # ---------------- Tab 3: Settings UI ----------------
    def _build_settings_tab(self):
        tab = self.tab_settings
        tab.grid_columnconfigure(1, weight=1)

        row = 0

        # Provider Selector
        ctk.CTkLabel(tab, text="Transcription Engine:", font=ctk.CTkFont(size=13, weight="bold")).grid(row=row, column=0, padx=12, pady=10, sticky="w")
        self.provider_combo = ctk.CTkComboBox(
            tab,
            values=["Deepgram Nova-2 (Recommended)", "Google Gemini (gemini-flash-lite)", "Groq Whisper", "OpenAI Whisper", "Simulated Test Mode (No Key Needed)"],
            command=self._on_provider_changed,
            width=360
        )
        self.provider_combo.grid(row=row, column=1, padx=12, pady=10, sticky="w")
        
        # Select current
        prov = self.config_mgr.get("provider", "deepgram")
        if prov == "deepgram":
            self.provider_combo.set("Deepgram Nova-2 (Recommended)")
        elif prov in ("gemini", "google"):
            self.provider_combo.set("Google Gemini (gemini-flash-lite)")
        elif prov == "groq":
            self.provider_combo.set("Groq Whisper")
        elif prov == "openai":
            self.provider_combo.set("OpenAI Whisper")
        else:
            self.provider_combo.set("Simulated Test Mode (No Key Needed)")

        row += 1

        ctk.CTkLabel(tab, text="Transcription Language:", font=ctk.CTkFont(size=13, weight="bold")).grid(row=row, column=0, padx=12, pady=10, sticky="w")
        self.language_combo = ctk.CTkComboBox(
            tab,
            values=list(SUPPORTED_LANGUAGES.values()),
            state="readonly",
            width=360
        )
        self.language_combo.grid(row=row, column=1, padx=12, pady=10, sticky="w")
        saved_language = self.config_mgr.get("transcription_language", "hi")
        self.language_combo.set(SUPPORTED_LANGUAGES.get(saved_language, SUPPORTED_LANGUAGES["auto"]))

        row += 1

        # Deepgram API Key
        ctk.CTkLabel(tab, text="Deepgram API Key:", font=ctk.CTkFont(size=12)).grid(row=row, column=0, padx=12, pady=6, sticky="w")
        self.deepgram_key_entry = ctk.CTkEntry(tab, show="•", width=360)
        self.deepgram_key_entry.grid(row=row, column=1, padx=12, pady=6, sticky="w")
        self.deepgram_key_entry.insert(0, self.config_mgr.get("deepgram_api_key", ""))

        row += 1

        # Google Gemini API Key
        ctk.CTkLabel(tab, text="Google Gemini API Key:", font=ctk.CTkFont(size=12)).grid(row=row, column=0, padx=12, pady=6, sticky="w")
        self.gemini_key_entry = ctk.CTkEntry(tab, show="•", width=360)
        self.gemini_key_entry.grid(row=row, column=1, padx=12, pady=6, sticky="w")
        self.gemini_key_entry.insert(0, self.config_mgr.get("gemini_api_key", ""))

        row += 1

        # Groq API Key
        ctk.CTkLabel(tab, text="Groq API Key:", font=ctk.CTkFont(size=12)).grid(row=row, column=0, padx=12, pady=6, sticky="w")
        self.groq_key_entry = ctk.CTkEntry(tab, show="•", width=360)
        self.groq_key_entry.grid(row=row, column=1, padx=12, pady=6, sticky="w")
        self.groq_key_entry.insert(0, self.config_mgr.get("groq_api_key", ""))

        row += 1

        # OpenAI API Key
        ctk.CTkLabel(tab, text="OpenAI API Key:", font=ctk.CTkFont(size=12)).grid(row=row, column=0, padx=12, pady=6, sticky="w")
        self.openai_key_entry = ctk.CTkEntry(tab, show="•", width=360)
        self.openai_key_entry.grid(row=row, column=1, padx=12, pady=6, sticky="w")
        self.openai_key_entry.insert(0, self.config_mgr.get("openai_api_key", ""))

        row += 1

        # Save Settings Button
        save_btn = ctk.CTkButton(
            tab,
            text="💾 Save Settings & API Keys",
            command=self._save_settings,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#059669",
            hover_color="#047857",
            height=34,
            width=220
        )
        save_btn.grid(row=row, column=1, padx=12, pady=16, sticky="w")

    # ---------------- UI Event Handlers ----------------
    def _populate_devices(self):
        devices = get_loopback_devices()
        device_names = ["Default System Playback Loopback"]
        for d in devices:
            device_names.append(d["name"])

        self.device_combo.configure(values=device_names)
        saved_device = self.config_mgr.get("selected_audio_device", "")
        if saved_device and saved_device in device_names:
            self.device_combo.set(saved_device)
            self.pipeline.set_device(saved_device)
        else:
            self.device_combo.set("Default System Playback Loopback")
            self.pipeline.set_device(None)

    def _on_device_selected(self, choice: str):
        if choice == "Default System Playback Loopback":
            self.pipeline.set_device(None)
            self.config_mgr.set("selected_audio_device", "")
        else:
            self.pipeline.set_device(choice)
            self.config_mgr.set("selected_audio_device", choice)
        self.config_mgr.save()
        self._set_footer(f"Selected audio loopback: {choice}")

    def _on_auto_fill_toggled(self):
        val = self.auto_fill_var.get()
        self.config_mgr.set("auto_fill_enabled", val)
        self.config_mgr.save()
        if val:
            self._set_footer("⚡ Auto-Fill ENABLED: Dialogue will automatically paste into the active focused window.")
        else:
            self._set_footer("Auto-Fill disabled: Dialogue will only appear in the transcript window.")

    def _on_topmost_toggled(self):
        val = self.topmost_var.get()
        self.attributes("-topmost", val)
        self.config_mgr.set("always_on_top", val)
        self.config_mgr.save()

    def _set_instruction_preset(self, preset_type: str):
        if preset_type == "clean":
            txt = DEFAULT_INSTRUCTIONS
        elif preset_type == "strict":
            txt = (
                "Full Verbatim. Transcribe every audible word exactly as spoken, including fillers, false starts, "
                "stutters, repetitions, slang, contractions, and speech errors. Do not correct grammar or paraphrase. "
                "Use double dashes for false starts or incomplete sentences. Label speakers consistently in bold. "
                "Do not invent timestamps or unintelligible markers; use those only when the audio and timing support them."
            )
        elif preset_type == "interview":
            txt = (
                "Clean Verbatim interview. Label speakers as Interviewer and Interviewee when their roles are clear; "
                "otherwise use consistent speaker numbers and mark uncertain identities with '?'. "
                "Remove only clearly non-meaningful fillers. Keep the speaker's grammar and wording; do not paraphrase."
            )
        else:
            txt = DEFAULT_INSTRUCTIONS

        self.instructions_box.delete("0.0", "end")
        self.instructions_box.insert("0.0", txt)
        self._save_instructions()

    def _save_instructions(self):
        text = self.instructions_box.get("0.0", "end").strip()
        self.config_mgr.set("custom_instructions", text)
        self.config_mgr.save()
        self._set_footer("Custom rules saved.")

    def _on_provider_changed(self, choice: str):
        if "Deepgram" in choice:
            prov = "deepgram"
        elif "Gemini" in choice:
            prov = "gemini"
        elif "Groq" in choice:
            prov = "groq"
        elif "OpenAI" in choice:
            prov = "openai"
        else:
            prov = "mock"

        self.config_mgr.set("provider", prov)
        self.config_mgr.save()
        self._update_engine_credentials()
        self._set_footer(f"Engine set to: {choice}")

    def _save_settings(self):
        self.config_mgr.set("deepgram_api_key", self.deepgram_key_entry.get().strip())
        self.config_mgr.set("gemini_api_key", self.gemini_key_entry.get().strip())
        self.config_mgr.set("groq_api_key", self.groq_key_entry.get().strip())
        self.config_mgr.set("openai_api_key", self.openai_key_entry.get().strip())
        selected_language = self.language_combo.get()
        language_code = next(
            (code for code, name in SUPPORTED_LANGUAGES.items() if name == selected_language),
            "auto",
        )
        self.config_mgr.set("transcription_language", language_code)
        self.config_mgr.save()
        self._update_engine_credentials()
        self._set_footer("All settings and API keys saved successfully.")
        messagebox.showinfo("Saved", "Settings and API Keys have been saved!")

    def _update_engine_credentials(self):
        prov = self.config_mgr.get("provider", "deepgram")
        api_key = self.config_mgr.get_api_key_for_provider(prov)
        self.transcriber.set_config(prov, api_key, self.config_mgr.get(f"{prov}_model", None))
        self.humanizer = FormattingHumanizer(
            llm_provider="gemini" if self.config_mgr.get("gemini_api_key") else prov,
            api_key=self.config_mgr.get("gemini_api_key") or api_key
        )
        if hasattr(self, "interview_helper"):
            self.interview_helper.set_config(
                provider="gemini" if self.config_mgr.get("gemini_api_key") else prov,
                api_key=self.config_mgr.get("gemini_api_key") or api_key,
                model_name=self.config_mgr.get("gemini_model", None)
            )

    # ---------------- Mode & Interview Event Handlers ----------------
    def _on_mode_changed(self, mode_str: str):
        if "Interview" in mode_str:
            self.config_mgr.set("app_mode", "interview_helper")
            self.tabview.set("💼 AI Interview Helper")
            self._set_footer("💼 AI Interview Helper Mode Active: Capturing questions and generating real-time answers.")
        else:
            self.config_mgr.set("app_mode", "transcriber")
            self.tabview.set("🎙️ Live Diarization")
            self._set_footer("🎙️ Transcriber Mode Active: Capturing audio and transcribing dialogue verbatim.")
        self.config_mgr.save()

    def _on_tab_changed(self):
        curr_tab = self.tabview.get()
        if curr_tab == "💼 AI Interview Helper":
            self.mode_selector.set("💼 AI Interview Helper")
            self.config_mgr.set("app_mode", "interview_helper")
            self.config_mgr.save()
            self._set_footer("💼 AI Interview Helper Mode Active.")
        elif curr_tab == "🎙️ Live Diarization":
            self.mode_selector.set("🎙️ Transcriber")
            self.config_mgr.set("app_mode", "transcriber")
            self.config_mgr.save()
            self._set_footer("🎙️ Transcriber Mode Active.")

    def _on_interview_style_changed(self, choice: str):
        self.config_mgr.set("interview_style", choice)
        self.config_mgr.save()
        self._set_footer(f"Interview response style set to: {choice}")

    def _on_auto_answer_toggled(self):
        val = self.auto_answer_var.get()
        self.config_mgr.set("interview_auto_answer", val)
        self.config_mgr.save()
        if val:
            self._set_footer("⚡ Auto-Answer ENABLED: Answers will be generated immediately after interviewer finishes.")
        else:
            self._set_footer("Auto-Answer disabled: Questions will be transcribed; click Regenerate or Answer to synthesize.")

    def _save_interview_profile(self):
        role = self.entry_interview_role.get().strip()
        context = self.entry_interview_context.get().strip()
        if role:
            self.config_mgr.set("interview_role", role)
        if context:
            self.config_mgr.set("interview_profile_context", context)
        self.config_mgr.save()
        self._set_footer("Target role and candidate background profile saved.")
        messagebox.showinfo("Profile Saved", "Your interview target role and profile details have been saved!")

    def _add_system_notice(self, text: str):
        card = ctk.CTkFrame(self.interview_chat_scroll, fg_color="#181d28", corner_radius=8)
        card.pack(fill="x", padx=10, pady=6)
        lbl = ctk.CTkLabel(
            card,
            text=text,
            font=ctk.CTkFont(size=12),
            text_color="#94a3b8",
            justify="left",
            wraplength=760
        )
        lbl.pack(padx=12, pady=10, anchor="w")

    def _add_interviewer_bubble(self, question_text: str, q_num: int):
        import time
        timestamp = time.strftime("%H:%M:%S")

        card = ctk.CTkFrame(self.interview_chat_scroll, fg_color="#1e293b", corner_radius=10)
        card.pack(fill="x", padx=10, pady=6)

        hdr_frame = ctk.CTkFrame(card, fg_color="transparent")
        hdr_frame.pack(fill="x", padx=12, pady=(10, 4))
        hdr_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            hdr_frame,
            text=f"👤 Interviewer  [{timestamp}]",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#60a5fa"
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            hdr_frame,
            text=f"Q#{q_num}",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#2563eb",
            text_color="white",
            corner_radius=6,
            padx=6,
            pady=1
        ).grid(row=0, column=1, sticky="e")

        body_lbl = ctk.CTkLabel(
            card,
            text=question_text,
            font=ctk.CTkFont(size=13),
            text_color="#f1f5f9",
            justify="left",
            wraplength=760
        )
        body_lbl.pack(padx=12, pady=(2, 10), anchor="w")
        self._scroll_chat_to_bottom()

    def _show_thinking_bubble(self):
        self._remove_thinking_bubble()
        self.thinking_card = ctk.CTkFrame(self.interview_chat_scroll, fg_color="#1a202c", corner_radius=8)
        self.thinking_card.pack(fill="x", padx=10, pady=4)
        lbl = ctk.CTkLabel(
            self.thinking_card,
            text="⚡ AI Chatbot is preparing response...",
            font=ctk.CTkFont(size=12, slant="italic"),
            text_color="#34d399"
        )
        lbl.pack(padx=12, pady=8, anchor="w")
        self._scroll_chat_to_bottom()

    def _remove_thinking_bubble(self):
        if self.thinking_card is not None:
            try:
                self.thinking_card.destroy()
            except Exception:
                pass
            self.thinking_card = None

    def _create_streaming_candidate_bubble(self):
        """
        Creates an active candidate response card in the chat and returns
        (append_token_ui, finalize_ui) for live streaming tokens.
        """
        self._remove_thinking_bubble()
        import time
        timestamp = time.strftime("%H:%M:%S")

        card = ctk.CTkFrame(
            self.interview_chat_scroll,
            fg_color="#064e3b",
            border_width=1,
            border_color="#10b981",
            corner_radius=10
        )
        card.pack(fill="x", padx=10, pady=6)

        hdr_frame = ctk.CTkFrame(card, fg_color="transparent")
        hdr_frame.pack(fill="x", padx=12, pady=(10, 4))
        hdr_frame.grid_columnconfigure(0, weight=1)

        status_lbl = ctk.CTkLabel(
            hdr_frame,
            text=f"🤖 Candidate Response  [{timestamp}]  ⚡ Streaming...",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#34d399"
        )
        status_lbl.grid(row=0, column=0, sticky="w")

        btn_frame = ctk.CTkFrame(hdr_frame, fg_color="transparent")
        btn_frame.grid(row=0, column=1, sticky="e")

        body_box = ctk.CTkTextbox(
            card,
            font=ctk.CTkFont(family="Consolas", size=13),
            wrap="word",
            fg_color="#04392b",
            text_color="#f8fafc",
            height=60,
            corner_radius=6
        )
        body_box.pack(fill="x", padx=12, pady=(2, 10))

        accumulated = [""]

        def append_token_ui(token: str):
            accumulated[0] += token
            body_box.insert("end", token)
            body_box.see("end")
            lines = max(3, min(24, accumulated[0].count("\n") + len(accumulated[0]) // 60 + 1))
            body_box.configure(height=lines * 22)
            self._scroll_chat_to_bottom()

        def finalize_ui(full_text: str):
            status_lbl.configure(text=f"🤖 Candidate Response  [{timestamp}]")
            btn_copy = ctk.CTkButton(
                btn_frame,
                text="📋 Copy",
                font=ctk.CTkFont(size=11),
                width=65,
                height=24,
                fg_color="#047857",
                hover_color="#065f46",
                command=lambda: self._copy_text(full_text)
            )
            btn_copy.pack(side="left", padx=2)

            btn_inject = ctk.CTkButton(
                btn_frame,
                text="⚡ Inject",
                font=ctk.CTkFont(size=11, weight="bold"),
                width=70,
                height=24,
                fg_color="#7c3aed",
                hover_color="#6d28d9",
                command=lambda: self._inject_specific_text(full_text)
            )
            btn_inject.pack(side="left", padx=2)
            self._scroll_chat_to_bottom()

        return append_token_ui, finalize_ui

    def _add_candidate_bubble(self, answer_text: str):
        append_ui, finalize_ui = self._create_streaming_candidate_bubble()
        append_ui(answer_text)
        finalize_ui(answer_text)

    def _scroll_chat_to_bottom(self):
        self.after(50, lambda: self._do_scroll())

    def _do_scroll(self):
        try:
            self.interview_chat_scroll._parent_canvas.yview_moveto(1.0)
        except Exception:
            pass

    def _copy_text(self, text: str):
        if text:
            import pyperclip
            pyperclip.copy(text)
            self._set_footer("Candidate response copied to clipboard.")

    def _inject_specific_text(self, text: str):
        if text:
            self._set_status_badge("INJECTING", "#7c3aed", "#ede9fe")
            ActiveWindowInjector.inject_text(text)
            self._set_footer("Injected candidate response into active window.")

    def _on_chat_input_submit(self):
        text = self.chat_input_entry.get().strip()
        if not text:
            return
        self.chat_input_entry.delete(0, "end")
        self.interview_question_count += 1
        self.last_interview_question = text

        # 1. Add Interviewer Question Bubble
        self._add_interviewer_bubble(text, self.interview_question_count)
        self._append_transcript(f"[Interview Q#{self.interview_question_count}] {text}")

        # 2. Create Candidate Streaming Bubble immediately
        append_token_ui, finalize_ui = self._create_streaming_candidate_bubble()
        self._set_status_badge("STREAMING", "#10b981", "#d1fae5")
        self._set_footer("⚡ Streaming candidate response...")

        def on_token_received(chunk: str):
            self.after(0, lambda c=chunk: append_token_ui(c))

        def worker():
            try:
                role = self.config_mgr.get("interview_role", DEFAULT_ROLE)
                profile = self.config_mgr.get("interview_profile_context", DEFAULT_PROFILE_CONTEXT)
                style = self.config_mgr.get("interview_style", "Direct Spoken Response (Only the Answer)")

                full_text = self.interview_helper.generate_interview_response_stream(
                    text,
                    role=role,
                    profile_context=profile,
                    style=style,
                    on_token=on_token_received
                )
                self.last_interview_answer = full_text
                self.last_formatted_text = full_text

                self.after(0, lambda: finalize_ui(full_text))
                self.after(0, lambda: self._set_status_badge("ANSWER READY", "#10b981", "#d1fae5"))
                self.after(0, lambda: self._set_footer("💡 Response ready! Press [Ctrl+Alt+V] to inject."))

                # Auto-fill if enabled
                if self.auto_fill_var.get() and full_text:
                    self.after(0, lambda: self._set_status_badge("INJECTING", "#7c3aed", "#ede9fe"))
                    ActiveWindowInjector.inject_text(full_text + "\n")
                    self.after(0, lambda: self._set_footer("⚡ Auto-filled response into active window!"))
            except Exception as e:
                self.after(0, lambda: finalize_ui(f"Error generating response: {e}"))
                self.after(0, lambda: self._set_footer(f"Error: {str(e)[:35]}"))

        threading.Thread(target=worker, daemon=True).start()

    def copy_interview_answer(self):
        ans = self.last_interview_answer
        if ans:
            import pyperclip
            pyperclip.copy(ans)
            self._set_footer("Last candidate response copied to clipboard.")
        else:
            self._set_footer("No response to copy yet.")

    def regenerate_interview_answer(self):
        q = self.last_interview_question
        if not q:
            self._set_footer("No question available to regenerate response.")
            return

        append_token_ui, finalize_ui = self._create_streaming_candidate_bubble()
        self._set_status_badge("STREAMING", "#10b981", "#d1fae5")
        self._set_footer("⚡ Regenerating candidate response...")

        def on_token_received(chunk: str):
            self.after(0, lambda c=chunk: append_token_ui(c))

        def worker():
            try:
                role = self.config_mgr.get("interview_role", DEFAULT_ROLE)
                profile = self.config_mgr.get("interview_profile_context", DEFAULT_PROFILE_CONTEXT)
                style = self.config_mgr.get("interview_style", "Direct Spoken Response (Only the Answer)")
                full_text = self.interview_helper.generate_interview_response_stream(
                    q, role=role, profile_context=profile, style=style, on_token=on_token_received
                )
                self.last_interview_answer = full_text
                self.last_formatted_text = full_text
                self.after(0, lambda: finalize_ui(full_text))
                self.after(0, lambda: self._set_status_badge("ANSWER READY", "#10b981", "#d1fae5"))
                self.after(0, lambda: self._set_footer("💡 Fresh candidate response ready!"))
            except Exception as e:
                self.after(0, lambda: finalize_ui(f"Error: {e}"))
                self.after(0, lambda: self._set_footer(f"Regeneration error: {str(e)[:35]}"))

        threading.Thread(target=worker, daemon=True).start()

    def clear_interview(self):
        for widget in self.interview_chat_scroll.winfo_children():
            try:
                widget.destroy()
            except Exception:
                pass
        self.thinking_card = None
        self._add_system_notice("💬 Interview chat cleared. Ready for the next question.")
        self.last_interview_question = ""
        self.last_interview_answer = ""
        self.interview_question_count = 0
        self.interview_helper.clear_history()
        self._set_footer("Interview chat history cleared.")

    def _test_loopback_capture(self):
        """Record 2 seconds of system sound and verify audio pipeline."""
        self._set_footer("Testing 2-second system loopback capture (play some audio now)...")
        self.status_badge.configure(text="● TESTING AUDIO", fg_color="#d97706", text_color="#fef3c7")

        def test_worker():
            try:
                selected = self.config_mgr.get("selected_audio_device", None)
                wav_bytes = AudioLoopbackPipeline.capture_test_sample(
                    duration_sec=2.0,
                    device_name=selected if selected else None
                )
                self.after(0, lambda: messagebox.showinfo(
                    "Loopback Capture Verified",
                    f"Successfully captured {len(wav_bytes)} bytes of digital system audio stream!\n"
                    "Your WASAPI loopback device is operational."
                ))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Loopback Test Failed", f"Could not capture loopback:\n{e}"))
            finally:
                self.after(0, lambda: self._on_status_change_update("IDLE"))

        threading.Thread(target=test_worker, daemon=True).start()

    # ---------------- Recording & Pipeline Control ----------------
    def toggle_recording(self):
        if self.pipeline.is_recording():
            self.pipeline.stop()
            self.btn_record.configure(
                text="▶ Start Listening (Ctrl+Alt+R)",
                fg_color="#059669",
                hover_color="#047857"
            )
            if hasattr(self, "btn_interview_record"):
                self.btn_interview_record.configure(
                    text="▶ Start Interview Capture (Ctrl+Alt+R)",
                    fg_color="#059669",
                    hover_color="#047857"
                )
            self._set_footer("Audio capture stopped.")
        else:
            self._update_engine_credentials()
            self.pipeline.start()
            self.btn_record.configure(
                text="⏹ Stop Listening (Ctrl+Alt+R)",
                fg_color="#dc2626",
                hover_color="#b91c1c"
            )
            if hasattr(self, "btn_interview_record"):
                self.btn_interview_record.configure(
                    text="⏹ Stop Interview Capture (Ctrl+Alt+R)",
                    fg_color="#dc2626",
                    hover_color="#b91c1c"
                )
            self._set_footer("Tapping into internal system audio loopback...")

    def _on_speech_chunk_received(self, wav_bytes: bytes, duration: float):
        """Invoked when VAD detects a complete speech utterance from system output."""
        self.after(0, lambda: self._set_status_badge("PROCESSING", "#2563eb", "#dbeafe"))
        self.after(0, lambda: self._set_footer(f"Transcribing {duration:.1f}s speech chunk..."))

        threading.Thread(target=self._process_chunk_worker, args=(wav_bytes,), daemon=True).start()

    def _process_chunk_worker(self, wav_bytes: bytes):
        custom_rules = self.config_mgr.get("custom_instructions", DEFAULT_INSTRUCTIONS)
        try:
            # 1. Speech Recognition & Diarization
            diarized_result: TranscriptionResult = self.transcriber.transcribe_and_diarize(
                wav_bytes,
                custom_instructions=custom_rules,
                language=self.config_mgr.get("transcription_language", "hi"),
            )
            raw_text = diarized_result.to_speaker_lines()

            # 2. Contextual Humanizer & Formatting Engine
            formatted_text = self.humanizer.format_transcript(
                raw_text,
                custom_instructions=custom_rules,
                use_llm=False
            )

            if not formatted_text or not formatted_text.strip():
                formatted_text = raw_text

            app_mode = self.config_mgr.get("app_mode", "transcriber")

            # ---------------- Case A: AI Interview Helper Mode ----------------
            if app_mode == "interview_helper":
                self.interview_question_count += 1
                q_text = formatted_text
                # Strip leading speaker prefix if present
                clean_q = q_text
                if ":" in clean_q:
                    _, clean_q = clean_q.split(":", 1)
                    clean_q = clean_q.strip()

                self.last_interview_question = clean_q or q_text
                
                # 1. Add Interviewer Question Bubble to Chat
                self.after(0, lambda: self._add_interviewer_bubble(clean_q or q_text, self.interview_question_count))
                self.after(0, lambda: self._append_transcript(f"[Interview Q#{self.interview_question_count}] {q_text}"))

                # 2. Auto-generate answer in the next chat message if enabled
                if self.auto_answer_var.get() and clean_q:
                    self.after(0, lambda: self._set_status_badge("STREAMING", "#10b981", "#d1fae5"))
                    self.after(0, lambda: self._set_footer("⚡ Streaming candidate response..."))

                    def launch_stream():
                        append_token_ui, finalize_ui = self._create_streaming_candidate_bubble()

                        def on_token_received(chunk: str):
                            self.after(0, lambda c=chunk: append_token_ui(c))

                        def stream_worker():
                            try:
                                role = self.config_mgr.get("interview_role", DEFAULT_ROLE)
                                profile = self.config_mgr.get("interview_profile_context", DEFAULT_PROFILE_CONTEXT)
                                style = self.config_mgr.get("interview_style", "Direct Spoken Response (Only the Answer)")

                                full_text = self.interview_helper.generate_interview_response_stream(
                                    clean_q,
                                    role=role,
                                    profile_context=profile,
                                    style=style,
                                    on_token=on_token_received
                                )
                                self.last_interview_answer = full_text
                                self.last_formatted_text = full_text

                                self.after(0, lambda: finalize_ui(full_text))
                                self.after(0, lambda: self._set_status_badge("ANSWER READY", "#10b981", "#d1fae5"))
                                self.after(0, lambda: self._set_footer("💡 Candidate response ready! Press [Ctrl+Alt+V] to inject."))

                                if self.auto_fill_var.get() and full_text:
                                    self.after(0, lambda: self._set_status_badge("INJECTING", "#7c3aed", "#ede9fe"))
                                    injected = ActiveWindowInjector.inject_text(full_text + "\n")
                                    if injected:
                                        self.after(0, lambda: self._set_footer("⚡ Auto-filled response into active window!"))
                            except Exception as e:
                                self.after(0, lambda: finalize_ui(f"Error: {e}"))
                                self.after(0, lambda: self._set_footer(f"Error: {str(e)[:35]}"))

                        threading.Thread(target=stream_worker, daemon=True).start()

                    self.after(0, launch_stream)

            # ---------------- Case B: Transcriber Mode ----------------
            else:
                self.last_formatted_text = formatted_text
                self.chunk_count += 1

                # Update GUI Transcript Box
                self.after(0, lambda: self._append_transcript(formatted_text))

                # Auto-Fill Injection into Active Foreground Window if enabled
                if self.auto_fill_var.get() and formatted_text:
                    self.after(0, lambda: self._set_status_badge("INJECTING", "#7c3aed", "#ede9fe"))
                    injected = ActiveWindowInjector.inject_text(formatted_text + "\n")
                    if injected:
                        self.after(0, lambda: self._set_footer("⚡ Auto-filled transcript into active window!"))
                    else:
                        self.after(0, lambda: self._set_footer("Auto-fill attempted."))

        except Exception as e:
            err_msg = f"[Error]: {str(e)}"
            print(f"[App] Pipeline processing error: {e}")
            self.after(0, lambda: self._append_transcript(err_msg))
            self.after(0, lambda: self._set_footer(f"Error: {str(e)[:40]}"))
        finally:
            if self.pipeline.is_recording():
                self.after(0, lambda: self._set_status_badge("LISTENING", "#059669", "#d1fae5"))
            else:
                self.after(0, lambda: self._set_status_badge("IDLE", "#374151", "#9ca3af"))

    def _append_transcript(self, text: str):
        current_content = self.transcript_box.get("0.0", "end")
        if "--- System Audio Diarizer Ready ---" in current_content:
            self.transcript_box.delete("0.0", "end")

        self.transcript_box.insert("end", f"{text}\n\n")
        self.transcript_box.see("end")

    def _on_audio_level_update(self, rms: float):
        """Update VU Meter (scaled 0.0 to 1.0)."""
        meter_val = min(1.0, rms * 15.0)
        self.after(0, lambda: self.vu_bar.set(meter_val))

    def _on_status_change_update(self, status: str):
        if status == "LISTENING":
            self.after(0, lambda: self._set_status_badge("LISTENING", "#059669", "#d1fae5"))
        elif status == "IDLE":
            self.after(0, lambda: self._set_status_badge("IDLE", "#374151", "#9ca3af"))
        elif "ERROR" in status:
            self.after(0, lambda: self._set_status_badge("ERROR", "#dc2626", "#fee2e2"))

    def _set_status_badge(self, text: str, bg_color: str, fg_color: str):
        self.status_badge.configure(text=f"● {text}", fg_color=bg_color, text_color=fg_color)

    def _set_footer(self, text: str):
        self.footer_status.configure(text=text)

    # ---------------- Actions ----------------
    def inject_active_window(self):
        """Manual injection hotkey/button."""
        app_mode = self.config_mgr.get("app_mode", "transcriber")
        if app_mode == "interview_helper" and self.last_interview_answer:
            text = self.last_interview_answer
        else:
            text = self.last_formatted_text

        if not text:
            if app_mode == "interview_helper":
                text = self.interview_a_box.get("0.0", "end").strip()
            else:
                box_text = self.transcript_box.get("0.0", "end").strip()
                if box_text and "--- System Audio Diarizer Ready ---" not in box_text:
                    lines = box_text.splitlines()
                    text = lines[-1] if lines else ""

        if text and "--- Real-time Suggested Answers" not in text and "--- System Audio Diarizer Ready" not in text:
            self._set_status_badge("INJECTING", "#7c3aed", "#ede9fe")
            ActiveWindowInjector.inject_text(text)
            self._set_footer("Injected text into active window.")
        else:
            self._set_footer("Nothing to inject yet.")

    def copy_transcript(self):
        text = self.transcript_box.get("0.0", "end").strip()
        if text:
            import pyperclip
            pyperclip.copy(text)
            self._set_footer("Transcript copied to clipboard.")

    def clear_transcript(self):
        self.transcript_box.delete("0.0", "end")
        self.last_formatted_text = ""
        self._set_footer("Transcript cleared.")

    def _on_close(self):
        self.pipeline.stop()
        self.hotkey_mgr.stop()
        self.destroy()


def run_app():
    app = AudioAgentApp()
    app.mainloop()


if __name__ == "__main__":
    run_app()
