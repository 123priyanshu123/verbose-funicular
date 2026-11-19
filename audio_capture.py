"""
Internal Audio Loopback Capture Pipeline using Windows WASAPI Loopback via soundcard.
Captures digital audio directly from the operating system's playback stream (earbuds,
speakers, or virtual audio outputs) without capturing room microphone noise.
"""

import io
import time
import wave
import threading
import numpy as np
import soundcard as sc
from typing import Callable, Optional, List, Dict


def get_loopback_devices() -> List[Dict[str, str]]:
    """Enumerate all available loopback capture devices on the system."""
    devices = []
    try:
        mics = sc.all_microphones(include_loopback=True)
        for mic in mics:
            if mic.isloopback:
                devices.append({
                    "id": mic.id,
                    "name": mic.name,
                    "is_loopback": True
                })
    except Exception as e:
        print(f"[AudioCapture] Error listing loopback devices: {e}")
    return devices


def get_default_loopback_device():
    """Retrieve the loopback microphone corresponding to the default playback speaker."""
    try:
        speaker = sc.default_speaker()
        if speaker is not None:
            return sc.get_microphone(speaker.name, include_loopback=True)
    except Exception as e:
        print(f"[AudioCapture] Error obtaining default speaker loopback: {e}")
    
    # Fallback to first available loopback mic
    mics = sc.all_microphones(include_loopback=True)
    for m in mics:
        if m.isloopback:
            return m
    return None


class AudioLoopbackPipeline:
    """
    Continuous background loopback audio capture with Voice Activity Detection (VAD),
    energy thresholding, silence segmentation, and WAV chunk dispatch.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        silence_threshold: float = 0.015,
        silence_timeout_sec: float = 1.5,
        min_speech_duration_sec: float = 0.25,
        max_chunk_duration_sec: float = 25.0,
        on_speech_chunk: Optional[Callable[[bytes, float], None]] = None,
        on_audio_level: Optional[Callable[[float], None]] = None,
        on_status_change: Optional[Callable[[str], None]] = None,
    ):
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_timeout_sec = silence_timeout_sec
        self.min_speech_duration_sec = min_speech_duration_sec
        self.max_chunk_duration_sec = max_chunk_duration_sec

        self.on_speech_chunk = on_speech_chunk
        self.on_audio_level = on_audio_level
        self.on_status_change = on_status_change

        self.selected_device_name: Optional[str] = None
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def set_device(self, device_name: Optional[str]):
        """Select a specific loopback device name or pass None for default."""
        self.selected_device_name = device_name

    def is_recording(self) -> bool:
        return self._is_running

    def start(self):
        """Start the loopback capture thread."""
        with self._lock:
            if self._is_running:
                return
            self._is_running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            if self.on_status_change:
                self.on_status_change("LISTENING")

    def stop(self, flush: bool = True):
        """Stop the loopback capture thread and optionally dispatch pending audio."""
        with self._lock:
            if not self._is_running:
                return
            self._is_running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        if self.on_status_change:
            self.on_status_change("IDLE")

    def _get_active_mic(self):
        """Resolve the soundcard loopback microphone object."""
        if self.selected_device_name:
            try:
                mic = sc.get_microphone(self.selected_device_name, include_loopback=True)
                if mic:
                    return mic
            except Exception:
                pass
        return get_default_loopback_device()

    def _capture_loop(self):
        """Internal audio loopback processing worker."""
        mic = self._get_active_mic()
        if not mic:
            print("[AudioCapture] No loopback audio device found.")
            if self.on_status_change:
                self.on_status_change("ERROR: No Loopback Device")
            self._is_running = False
            return

        print(f"[AudioCapture] Tapping into loopback device: {mic.name}")

        # Block size: 100 ms of audio per iteration
        block_frames = int(self.sample_rate * 0.1)
        
        # Audio accumulator
        speech_frames: List[np.ndarray] = []
        is_speech_active = False
        silence_start_time: Optional[float] = None
        speech_start_time: Optional[float] = None

        # Pre-roll ring buffer (0.3s) to avoid clipping beginning of utterances
        pre_roll_limit = 5
        pre_roll_buffer: List[np.ndarray] = []

        try:
            with mic.recorder(samplerate=self.sample_rate, channels=None) as recorder:
                while self._is_running:
                    # Capture one block
                    data = recorder.record(numframes=block_frames)
                    if data is None or len(data) == 0:
                        time.sleep(0.01)
                        continue

                    mono_data = self._downmix_audio(data)

                    # Calculate RMS energy for VAD and VU meter
                    rms = float(np.sqrt(np.mean(mono_data ** 2)))
                    if self.on_audio_level:
                        self.on_audio_level(rms)

                    # Manage pre-roll buffer
                    pre_roll_buffer.append(mono_data)
                    if len(pre_roll_buffer) > pre_roll_limit:
                        pre_roll_buffer.pop(0)

                    now = time.time()

                    # Voice Activity Detection Logic
                    if rms >= self.silence_threshold:
                        if not is_speech_active:
                            # Speech just began: prepend pre-roll buffer
                            is_speech_active = True
                            speech_start_time = now
                            speech_frames = list(pre_roll_buffer)
                        else:
                            speech_frames.append(mono_data)
                        silence_start_time = None
                    else:
                        if is_speech_active:
                            speech_frames.append(mono_data)
                            if silence_start_time is None:
                                silence_start_time = now
                            elif (now - silence_start_time) >= self.silence_timeout_sec:
                                # Silence threshold exceeded: finalize chunk
                                self._dispatch_chunk(speech_frames)
                                speech_frames = []
                                is_speech_active = False
                                silence_start_time = None
                                speech_start_time = None

                    # Max chunk duration safety limit
                    if is_speech_active and speech_start_time and (now - speech_start_time) >= self.max_chunk_duration_sec:
                        self._dispatch_chunk(speech_frames)
                        speech_frames = []
                        is_speech_active = False
                        silence_start_time = None
                        speech_start_time = None

        except Exception as e:
            print(f"[AudioCapture] Recorder exception: {e}")
            if self.on_status_change:
                self.on_status_change(f"ERROR: {str(e)[:30]}")
        finally:
            # Flush any remaining frames on stop
            if speech_frames:
                self._dispatch_chunk(speech_frames)

    def _dispatch_chunk(self, frames: List[np.ndarray]):
        """Convert float32 mono audio frames to 16-bit PCM WAV and trigger callback."""
        if not frames:
            return
        
        full_audio = np.concatenate(frames)
        total_duration = len(full_audio) / self.sample_rate

        if total_duration < self.min_speech_duration_sec:
            # Drop very brief click/pop artifacts
            return

        # Normalize and convert to 16-bit integer PCM
        clipped = np.clip(full_audio, -1.0, 1.0)
        pcm_data = (clipped * 32767).astype(np.int16)

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm_data.tobytes())

        wav_bytes = wav_buffer.getvalue()
        print(f"[AudioCapture] Speech chunk ready: {total_duration:.2f}s, {len(wav_bytes)} bytes")

        if self.on_speech_chunk:
            self.on_speech_chunk(wav_bytes, total_duration)

    @staticmethod
    def _downmix_audio(data: np.ndarray) -> np.ndarray:
        """Downmix stereo while avoiding cancellation from out-of-phase channels."""
        if data.ndim == 1 or data.shape[1] < 2:
            return data if data.ndim == 1 else data[:, 0]

        left = data[:, 0]
        right = data[:, 1]
        correlation = np.corrcoef(left, right)[0, 1]
        if np.isfinite(correlation) and correlation < -0.5:
            left_rms = float(np.sqrt(np.mean(left ** 2)))
            right_rms = float(np.sqrt(np.mean(right ** 2)))
            return left if left_rms >= right_rms else right

        return np.mean(data, axis=1)

    @staticmethod
    def capture_test_sample(duration_sec: float = 3.0, sample_rate: int = 16000, device_name: Optional[str] = None) -> bytes:
        """Capture a short test loopback snippet to verify system audio capture."""
        if device_name:
            mic = sc.get_microphone(device_name, include_loopback=True)
        else:
            mic = get_default_loopback_device()

        if not mic:
            raise RuntimeError("No loopback device found.")

        num_frames = int(sample_rate * duration_sec)
        with mic.recorder(samplerate=sample_rate, channels=None) as recorder:
            data = recorder.record(numframes=num_frames)

        mono = AudioLoopbackPipeline._downmix_audio(data)

        clipped = np.clip(mono, -1.0, 1.0)
        pcm = (clipped * 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()
