#!/usr/bin/env python3
"""
Dynamic Real-time Audio Source Detector for Aggregate Device
Shows which channels are active and infers device names based on channel activity patterns
"""

import pyaudio
import numpy as np
import sys
import signal

# Configuration
CHUNK_SIZE = 1024
SAMPLE_RATE = 16000
RMS_THRESHOLD = 20.0  # Adjust this if needed - lower = more sensitive

def get_available_input_devices(p):
    """Get all available input devices with their channel counts"""
    device_count = p.get_device_count()
    input_devices = []
    
    for i in range(device_count):
        try:
            dev_info = p.get_device_info_by_index(i)
            dev_name = dev_info.get('name', '')
            dev_input_channels = dev_info.get('maxInputChannels', 0)
            
            # Skip output-only devices
            if dev_input_channels > 0:
                input_devices.append({
                    'index': i,
                    'name': dev_name,
                    'channels': dev_input_channels
                })
        except:
            continue
    
    return input_devices

def detect_channel_mapping(p, aggregate_device_index):
    """
    Show available devices and their channel counts
    Note: PyAudio doesn't tell us which channels in Aggregate Device map to which subdevice
    We can only infer based on typical Aggregate Device setup
    """
    device_info = p.get_device_info_by_index(aggregate_device_index)
    device_name = device_info.get('name', '')
    total_channels = device_info.get('maxInputChannels', 0)
    
    print(f"Aggregate Device: {device_name}")
    print(f"Total channels: {total_channels}")
    print()
    
    # Get all available input devices
    input_devices = get_available_input_devices(p)
    
    print("Available Input Devices:")
    print("-" * 80)
    for dev in input_devices:
        if dev['index'] != aggregate_device_index:  # Don't show the aggregate device itself
            is_blackhole = 'blackhole' in dev['name'].lower()
            device_type = "🔊 System Audio" if is_blackhole else "🎤 User Input"
            print(f"  [{dev['index']}] {dev['name']}")
            print(f"      Type: {device_type}")
            print(f"      Input Channels: {dev['channels']}")
            print()
    
    print("-" * 80)
    print()
    print("⚠️  Important Note:")
    print("   PyAudio doesn't expose which channels in the Aggregate Device")
    print("   correspond to which subdevice. The channel mapping is configured")
    print("   in macOS Audio MIDI Setup when creating the Aggregate Device.")
    print()
    print("   We can detect which channels have audio activity, but we need to")
    print("   infer which device it's from based on:")
    print("   1. Channel activity patterns")
    print("   2. Typical Aggregate Device setup (BlackHole first, then mics)")
    print()
    
    # Create a simple mapping: we'll infer based on typical setup
    # BlackHole devices are usually added first, then microphones
    blackhole_devices = [d for d in input_devices if d['index'] != aggregate_device_index and 'blackhole' in d['name'].lower()]
    other_devices = [d for d in input_devices if d['index'] != aggregate_device_index and 'blackhole' not in d['name'].lower()]
    
    # Estimate channel ranges (this is an inference, not exact)
    current_channel = 0
    estimated_mapping = {}
    
    print("Estimated Channel Mapping (based on typical Aggregate Device setup):")
    print("-" * 80)
    
    # BlackHole devices typically come first
    for dev in blackhole_devices:
        channels = list(range(current_channel, current_channel + dev['channels']))
        for ch in channels:
            if ch < total_channels:
                estimated_mapping[ch] = {
                    'device_name': dev['name'],
                    'device_type': 'System Audio',
                    'device_index': dev['index']
                }
        print(f"  Channels {[ch+1 for ch in channels if ch < total_channels]} (0-indexed: {[ch for ch in channels if ch < total_channels]}): {dev['name']} (🔊 System Audio)")
        current_channel += dev['channels']
    
    # Other devices (mics, earphones) come after
    for dev in other_devices:
        channels = list(range(current_channel, current_channel + dev['channels']))
        for ch in channels:
            if ch < total_channels:
                estimated_mapping[ch] = {
                    'device_name': dev['name'],
                    'device_type': 'User Input',
                    'device_index': dev['index']
                }
        print(f"  Channels {[ch+1 for ch in channels if ch < total_channels]} (0-indexed: {[ch for ch in channels if ch < total_channels]}): {dev['name']} (🎤 User Input)")
        current_channel += dev['channels']
    
    # Handle remaining channels
    if current_channel < total_channels:
        remaining = list(range(current_channel, total_channels))
        for ch in remaining:
            estimated_mapping[ch] = {
                'device_name': 'Unknown Device',
                'device_type': 'Unknown',
                'device_index': None
            }
        print(f"  Channels {[ch+1 for ch in remaining]} (0-indexed: {remaining}): Unknown Device")
    
    print()
    print("⚠️  This mapping is an ESTIMATE based on typical Aggregate Device configuration.")
    print("   The actual mapping depends on how you configured it in Audio MIDI Setup.")
    print()
    
    return estimated_mapping

def detect_audio_source(audio_data, num_channels, channel_mapping):
    """Analyze audio data and determine which device(s) are active"""
    # Convert bytes to numpy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    
    # Reshape to separate channels
    num_samples = len(audio_array) // num_channels
    if num_samples * num_channels != len(audio_array):
        return None, None
    
    audio_array = audio_array.reshape(num_samples, num_channels)
    
    # Calculate RMS per channel
    audio_float = audio_array.astype(np.float32)
    rms_per_channel = np.sqrt(np.mean(audio_float ** 2, axis=0))
    
    # Find active channels
    active_channels = []
    for i, rms in enumerate(rms_per_channel):
        if rms > RMS_THRESHOLD:
            active_channels.append(i)
    
    if not active_channels:
        return None, None
    
    # Group active channels by device
    device_activity = {}  # device_name -> {'channels': [], 'type': '', 'device_index': None}
    
    for ch in active_channels:
        if ch in channel_mapping:
            device_info = channel_mapping[ch]
            device_name = device_info['device_name']
            
            if device_name not in device_activity:
                device_activity[device_name] = {
                    'channels': [],
                    'type': device_info['device_type'],
                    'device_index': device_info['device_index']
                }
            
            device_activity[device_name]['channels'].append(ch)
    
    # Build source description
    sources = []
    for device_name, info in device_activity.items():
        if info['type'] == 'System Audio':
            sources.append(f"🔊 {device_name} (System Audio)")
        elif info['type'] == 'User Input':
            sources.append(f"🎤 {device_name} (User Input)")
        else:
            sources.append(f"❓ {device_name}")
    
    source = " + ".join(sources) if len(sources) > 1 else sources[0]
    
    return source, active_channels, device_activity

def main():
    print("=" * 80)
    print("Dynamic Real-time Audio Source Detector")
    print("=" * 80)
    print()
    
    p = None
    stream = None
    
    def signal_handler(sig, frame):
        print("\n\nStopping...")
        if stream:
            stream.stop_stream()
            stream.close()
        if p:
            p.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        p = pyaudio.PyAudio()
        
        # Get default input device
        default_input = p.get_default_input_device_info()
        device_index = default_input['index']
        device_name = default_input['name']
        max_channels = default_input['maxInputChannels']
        
        print(f"Using device: {device_name} (Index: {device_index})")
        print(f"Max input channels: {max_channels}")
        print()
        
        # Get channel mapping (estimated)
        channel_mapping = detect_channel_mapping(p, device_index)
        
        # Determine number of channels to use
        num_channels = min(max_channels, 18) if max_channels > 1 else 1
        
        print(f"Opening stream with {num_channels} channels...")
        print("Listening for audio activity...")
        print("Press Ctrl+C to stop")
        print()
        print("-" * 80)
        print()
        
        # Open stream
        stream = p.open(
            format=pyaudio.paInt16,
            channels=num_channels,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            input_device_index=device_index
        )
        
        stream.start_stream()
        print("✓ Stream started. Listening...")
        print()
        
        last_source = None
        silence_count = 0
        
        while True:
            try:
                # Read audio data
                audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                
                # Detect source
                result = detect_audio_source(audio_data, num_channels, channel_mapping)
                
                if result and result[0]:
                    source, active_channels, device_activity = result
                    
                    if source != last_source:
                        print(f"[SOURCE] {source}")
                        
                        # Show detailed channel information
                        for device_name, info in device_activity.items():
                            channels = info['channels']
                            device_type = info['type']
                            print(f"         {device_name} ({device_type}): Channels {channels} (UI: {[ch+1 for ch in channels]})")
                        
                        print()
                    last_source = source
                    silence_count = 0
                else:
                    silence_count += 1
                    if silence_count == 50:  # After ~3 seconds of silence, reset
                        if last_source:
                            print("[SILENCE] No audio detected")
                            print()
                        last_source = None
                        silence_count = 0
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error reading audio: {e}")
                continue
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        if p:
            p.terminate()
        print("\nStopped.")

if __name__ == '__main__':
    main()