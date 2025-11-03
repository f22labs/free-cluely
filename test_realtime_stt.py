#!/usr/bin/env python3
"""Quick test to verify RealtimeSTT is working"""

import sys
import json
from RealtimeSTT import AudioToTextRecorder

print("Testing RealtimeSTT...", file=sys.stderr)
sys.stderr.flush()

# Simple test with minimal config
recorder_config = {
    'spinner': False,
    'model': 'tiny.en',  # Use tiny for faster testing
    'realtime_model_type': 'tiny.en',
    'language': 'en',
    'enable_realtime_transcription': True,
    'no_log_file': True,
}

print("Creating recorder...", file=sys.stderr)
sys.stderr.flush()

try:
    recorder = AudioToTextRecorder(**recorder_config)
    print("Recorder created successfully!", file=sys.stderr)
    sys.stderr.flush()
    
    def on_realtime_update(text):
        if text:
            print(f"[REALTIME] {text}", file=sys.stderr)
            sys.stderr.flush()
    
    def on_complete(text):
        if text:
            print(f"[COMPLETE] {text}", file=sys.stderr)
            sys.stderr.flush()
    
    # Test with callbacks
    print("Starting transcription test... Speak something!", file=sys.stderr)
    sys.stderr.flush()
    
    # Try one transcription
    print("Waiting for speech...", file=sys.stderr)
    sys.stderr.flush()
    text = recorder.text(on_complete)
    
    if text:
        print(f"SUCCESS! Got transcription: {text}")
    else:
        print("WARNING: recorder.text() returned empty")
        
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    import traceback
    print(traceback.format_exc(), file=sys.stderr)
    sys.exit(1)

