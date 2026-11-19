"""
Active Window Injector & Global Hotkey Manager.
Handles hands-free injection into focused form fields, test inputs, or chat apps via
clipboard emulation, and registers OS-wide hotkeys (Ctrl+Alt+R, Ctrl+Alt+V).
"""

import time
import threading
import pyperclip
import pyautogui
from typing import Optional, Callable


class ActiveWindowInjector:
    """Safely injects text into the foreground active application."""

    @staticmethod
    def inject_text(text: str, delay_sec: float = 0.05, restore_clipboard: bool = False) -> bool:
        """
        Copy text to system clipboard and simulate Ctrl+V to paste into active focused control.
        """
        if not text or not text.strip():
            return False

        try:
            previous_clip = ""
            if restore_clipboard:
                try:
                    previous_clip = pyperclip.paste()
                except Exception:
                    pass

            # Write text to clipboard
            pyperclip.copy(text)
            time.sleep(delay_sec)

            # Emulate paste command
            pyautogui.hotkey("ctrl", "v")

            if restore_clipboard and previous_clip:
                time.sleep(0.1)
                try:
                    pyperclip.copy(previous_clip)
                except Exception:
                    pass

            return True
        except Exception as e:
            print(f"[Injector] Failed to inject text: {e}")
            return False


class HotkeyManager:
    """Manages global OS hotkeys using 'keyboard' or 'pynput' as fallback."""

    def __init__(
        self,
        on_toggle_recording: Optional[Callable[[], None]] = None,
        on_inject_last: Optional[Callable[[], None]] = None,
        on_clear_transcript: Optional[Callable[[], None]] = None,
    ):
        self.on_toggle_recording = on_toggle_recording
        self.on_inject_last = on_inject_last
        self.on_clear_transcript = on_clear_transcript

        self._keyboard_registered = False
        self._pynput_listener = None

    def start(self):
        """Register global hotkeys in the background."""
        # Try keyboard library first
        try:
            import keyboard
            if self.on_toggle_recording:
                keyboard.add_hotkey("ctrl+alt+r", self._safe_call(self.on_toggle_recording))
            if self.on_inject_last:
                keyboard.add_hotkey("ctrl+alt+v", self._safe_call(self.on_inject_last))
            if self.on_clear_transcript:
                keyboard.add_hotkey("ctrl+alt+c", self._safe_call(self.on_clear_transcript))
            self._keyboard_registered = True
            print("[HotkeyManager] Global hotkeys registered via 'keyboard' (Ctrl+Alt+R, Ctrl+Alt+V, Ctrl+Alt+C)")
            return
        except Exception as e:
            print(f"[HotkeyManager] 'keyboard' registration failed ({e}), attempting pynput fallback...")

        # Fallback to pynput
        try:
            from pynput import keyboard as pk
            hotkeys = {}
            if self.on_toggle_recording:
                hotkeys['<ctrl>+<alt>+r'] = self._safe_call(self.on_toggle_recording)
            if self.on_inject_last:
                hotkeys['<ctrl>+<alt>+v'] = self._safe_call(self.on_inject_last)
            if self.on_clear_transcript:
                hotkeys['<ctrl>+<alt>+c'] = self._safe_call(self.on_clear_transcript)

            self._pynput_listener = pk.GlobalHotKeys(hotkeys)
            self._pynput_listener.start()
            print("[HotkeyManager] Global hotkeys registered via 'pynput'")
        except Exception as e:
            print(f"[HotkeyManager] Failed to start hotkey listener: {e}")

    def stop(self):
        """Unregister hotkeys."""
        if self._keyboard_registered:
            try:
                import keyboard
                keyboard.unhook_all_hotkeys()
            except Exception:
                pass
            self._keyboard_registered = False

        if self._pynput_listener:
            try:
                self._pynput_listener.stop()
            except Exception:
                pass
            self._pynput_listener = None

    def _safe_call(self, callback: Callable[[], None]):
        """Wraps callback in non-blocking thread to avoid freezing OS input thread."""
        def wrapper():
            threading.Thread(target=callback, daemon=True).start()
        return wrapper
