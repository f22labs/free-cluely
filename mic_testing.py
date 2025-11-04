#!/usr/bin/env python3
"""
Microphone Test Script
Records audio for 10 seconds, plays it back, and prompts user for confirmation.
"""

def test_microphone():
    """Test microphone by recording for 10 seconds, playing back, and asking user"""
    try:
        import pyaudio
        import time
        import struct
        
        # Audio parameters
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK = 1024
        RECORD_SECONDS = 10
        
        p = pyaudio.PyAudio()
        
        # Get default input device info
        try:
            default_input = p.get_default_input_device_info()
            print(f"\n[AUDIO TEST] Using microphone: {default_input['name']}")
            print(f"[AUDIO TEST] Sample rate: {default_input['defaultSampleRate']} Hz")
            print(f"[AUDIO TEST] Max input channels: {default_input['maxInputChannels']}")
        except Exception as e:
            print(f"[AUDIO TEST] Warning: Could not get device info: {e}")
        
        # Open input stream for recording
        print(f"\n[AUDIO TEST] Opening microphone for recording...")
        input_stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        print(f"[AUDIO TEST] ✓ Microphone stream opened successfully!")
        print(f"\n[AUDIO TEST] ⏺️  Recording for {RECORD_SECONDS} seconds...")
        print(f"[AUDIO TEST] Please speak now! (Recording in progress...)")
        
        # Record audio
        frames = []
        max_amplitude = 0
        total_chunks = int(RATE / CHUNK * RECORD_SECONDS)
        
        for i in range(total_chunks):
            data = input_stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            
            # Check audio level
            audio_data = struct.unpack('<' + ('h' * (len(data) // 2)), data)
            chunk_max = max(abs(x) for x in audio_data)
            max_amplitude = max(max_amplitude, chunk_max)
            
            # Show progress
            elapsed = (i + 1) / (RATE / CHUNK)
            if i % (int(RATE / CHUNK)) == 0:  # Every second
                print(f"[AUDIO TEST] Recording... {elapsed:.1f}s / {RECORD_SECONDS}s (max amplitude: {max_amplitude})")
        
        input_stream.stop_stream()
        input_stream.close()
        
        print(f"\n[AUDIO TEST] ✓ Recording complete!")
        print(f"[AUDIO TEST] Max audio amplitude detected: {max_amplitude}")
        
        if max_amplitude < 100:
            print(f"[AUDIO TEST] ⚠️  WARNING: Very low audio levels detected (max: {max_amplitude})")
            print(f"[AUDIO TEST] ⚠️  This might indicate microphone is not picking up audio properly")
        
        # Prepare audio data for playback
        audio_data_bytes = b''.join(frames)
        
        print(f"\n[AUDIO TEST] 🔊 Playing back recorded audio...")
        time.sleep(0.5)  # Brief pause before playback
        
        # Open output stream for playback
        output_stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True,
            frames_per_buffer=CHUNK
        )
        
        # Play back audio
        print(f"[AUDIO TEST] Playing {RECORD_SECONDS} seconds of recorded audio...")
        for i in range(0, len(audio_data_bytes), CHUNK):
            chunk = audio_data_bytes[i:i + CHUNK]
            output_stream.write(chunk)
            
            # Show progress
            elapsed = i / (RATE * 2)  # 2 bytes per sample (16-bit)
            if i % (RATE * 2) < CHUNK * 2:  # Every second
                print(f"[AUDIO TEST] Playing... {elapsed:.1f}s / {RECORD_SECONDS}s")
        
        output_stream.stop_stream()
        output_stream.close()
        
        print(f"\n[AUDIO TEST] ✓ Playback complete!")
        
        # Prompt user for confirmation
        print(f"\n" + "="*60)
        print(f"[AUDIO TEST] Did you hear the audio playback clearly?")
        print(f"[AUDIO TEST] Please answer 'yes' or 'no'")
        print(f"="*60)
        
        response = input("\nYour answer: ").strip().lower()
        
        p.terminate()
        
        # Evaluate response
        if response in ['yes', 'y']:
            print(f"\n[AUDIO TEST] ✓✓✓ SUCCESS! Microphone test PASSED ✓✓✓")
            print(f"[AUDIO TEST] Your microphone is working correctly!")
            return True
        elif response in ['no', 'n']:
            print(f"\n[AUDIO TEST] ✗✗✗ FAILED! Microphone test FAILED ✗✗✗")
            print(f"[AUDIO TEST] You did not hear the audio playback clearly.")
            print(f"[AUDIO TEST] This might indicate:")
            print(f"[AUDIO TEST]   - Microphone is not recording properly")
            print(f"[AUDIO TEST]   - Speakers/headphones are not working")
            print(f"[AUDIO TEST]   - Audio output device is incorrect")
            return False
        else:
            print(f"\n[AUDIO TEST] ⚠️  Invalid response. Please run the test again.")
            return False
        
    except Exception as e:
        print(f"\n[AUDIO TEST] ✗✗✗ ERROR: Microphone test FAILED ✗✗✗")
        print(f"[AUDIO TEST] Error: {e}")
        import traceback
        print(f"\n[AUDIO TEST] Traceback:")
        print(traceback.format_exc())
        return False


if __name__ == '__main__':
    print("="*60)
    print("MICROPHONE TEST - Record & Playback")
    print("="*60)
    test_microphone()