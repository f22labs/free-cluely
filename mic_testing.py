# Add this as a new method in RealtimeSTTService class
def test_microphone():
    """Test if microphone is accessible"""
    try:
        import pyaudio
        import wave
        import time
        
        p = pyaudio.PyAudio()
        
        # Try to open a stream
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )
        
        print("[AUDIO DEBUG] Microphone stream opened successfully!")
        print("[AUDIO DEBUG] Reading 1 second of audio to test...")
        
        # Read some audio
        frames = []
        for _ in range(0, int(16000 / 1024)):  # 1 second
            data = stream.read(1024)
            frames.append(data)
            # Check if we're getting non-zero data
            import struct
            audio_data = struct.unpack('<' + ('h' * (len(data) // 2)), data)
            max_amplitude = max(abs(x) for x in audio_data)
            if max_amplitude > 100:  # Threshold for actual audio
                print(f"[AUDIO DEBUG] Audio detected! Max amplitude: {max_amplitude}")
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        print("[AUDIO DEBUG] Microphone test PASSED - audio is accessible")
        return True
        
    except Exception as e:
        print(f"[AUDIO DEBUG] Microphone test FAILED: {e}")
        import traceback
        print(f"[AUDIO DEBUG] Traceback: {traceback.format_exc()}")
        return False

test_microphone()