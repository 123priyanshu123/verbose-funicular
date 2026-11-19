"""
AI Audio Loopback Transcriber & Diarization Agent
Main entrypoint supporting Desktop Overlay GUI and CLI inspection modes.
"""

import sys
import argparse
from audio_capture import get_loopback_devices, get_default_loopback_device, AudioLoopbackPipeline


def list_devices():
    print("\n--- Available Windows WASAPI Loopback Capture Devices ---")
    devices = get_loopback_devices()
    if not devices:
        print("No loopback devices detected. Ensure Windows audio services are active.")
        return

    default_device = get_default_loopback_device()
    default_name = default_device.name if default_device else "None"

    for i, dev in enumerate(devices, 1):
        is_default = " (Default Speaker Loopback)" if dev["name"] == default_name else ""
        print(f"[{i}] {dev['name']}{is_default}")
    print("--------------------------------------------------------\n")


def test_capture():
    print("Capturing 2 seconds of system audio loopback to verify WASAPI...")
    try:
        sample = AudioLoopbackPipeline.capture_test_sample(duration_sec=2.0)
        print(f"Success! Captured {len(sample)} bytes of raw PCM WAV stream.")
        print("Your internal loopback pipeline is operational and ready to capture meeting/video sound.")
    except Exception as e:
        print(f"Error testing loopback capture: {e}")


def main():
    parser = argparse.ArgumentParser(description="Internal Audio Loopback Pipeline & Diarization Agent")
    parser.add_argument("--list-devices", action="store_true", help="List detected WASAPI loopback devices")
    parser.add_argument("--test", action="store_true", help="Run 2-second loopback capture test")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.test:
        test_capture()
        return

    # Launch GUI
    from gui import run_app
    run_app()


if __name__ == "__main__":
    main()
