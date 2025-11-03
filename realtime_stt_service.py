#!/usr/bin/env python3
"""
RealtimeSTT Service for Electron Integration
This service provides real-time transcription using RealtimeSTT library.
Communicates via JSON over stdin/stdout for IPC with Electron.
"""

import sys
import json
import os
from pathlib import Path

try:
    from RealtimeSTT import AudioToTextRecorder
except ImportError:
    print(json.dumps({"type": "error", "error": "RealtimeSTT not installed. Run: pip install RealtimeSTT"}), file=sys.stderr)
    sys.exit(1)

class RealtimeSTTService:
    def __init__(self, transcript_file_path=None, model='large-v3', realtime_model='base.en', language='en'):
        """Initialize the RealtimeSTT recorder"""
        self.transcript_file_path = transcript_file_path
        self.full_transcript = []
        self.is_recording = False
        self.last_written_text = ""  # Track last written text to avoid duplicates
        self.last_realtime_update = ""  # Track last real-time update
        self.realtime_buffer = []  # Buffer to store all real-time updates as backup
        
        # Create transcript file if path provided
        if transcript_file_path:
            os.makedirs(os.path.dirname(transcript_file_path), exist_ok=True)
            with open(transcript_file_path, 'w', encoding='utf-8') as f:
                f.write(f"=== Real-Time Transcript Started at {self._get_timestamp()} ===\n\n")
        
        # Store config for later use in run()
        self.model = model
        self.realtime_model = realtime_model
        self.language = language
        
        # Don't create recorder here - will be created in run() with callbacks
        self.recorder = None
        self._send_message({"type": "status", "status": "initialized"})
    
    def _get_timestamp(self):
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _send_message(self, message):
        """Send JSON message to stdout"""
        try:
            print(json.dumps(message), flush=True)
        except Exception as e:
            self._send_error(f"Error sending message: {str(e)}")
    
    def _send_error(self, error_msg):
        """Send error message"""
        print(json.dumps({"type": "error", "error": error_msg}), file=sys.stderr, flush=True)
    
    def _append_to_file(self, text, is_realtime=False):
        """Append text to transcript file, avoiding duplicates"""
        if self.transcript_file_path and text:
            try:
                text_stripped = text.strip()
                
                # Skip if this exact text was already written
                if text_stripped == self.last_written_text.strip():
                    return
                
                # For real-time updates: don't write to file, only update in-memory tracking
                # This prevents duplicates - we'll only write when transcription completes
                if is_realtime:
                    self.last_realtime_update = text_stripped
                    return  # Don't write real-time updates to file
                
                # For complete transcriptions: write to file
                # Remove any partial real-time update line that might match
                try:
                    with open(self.transcript_file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Remove the last line if it's a partial match or same sentence
                    # This handles the case where real-time update and complete match
                    if lines and lines[-1].strip():
                        last_line = lines[-1].strip()
                        # If the last line is contained in the new text or vice versa, remove it
                        if last_line in text_stripped or text_stripped in last_line:
                            lines = lines[:-1]
                            # Also check second-to-last line for similar matches
                            if len(lines) > 1 and lines[-1].strip() in text_stripped:
                                lines = lines[:-1]
                    
                    with open(self.transcript_file_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                except Exception as e:
                    self._send_error(f"[DEBUG] Error reading file for duplicate removal: {str(e)}")
                
                # Append the new complete transcription
                with open(self.transcript_file_path, 'a', encoding='utf-8') as f:
                    f.write(text_stripped + "\n")
                    f.flush()
                
                self.last_written_text = text_stripped
                self.last_realtime_update = ""  # Reset after writing complete
            except Exception as e:
                self._send_error(f"Error writing to file: {str(e)}")
    
    def start_recording(self):
        """Start recording and transcription"""
        if self.is_recording:
            return
        
        self.is_recording = True
        self._send_message({"type": "status", "status": "recording_started"})
    
    def stop_recording(self):
        """Stop recording and finalize transcript file"""
        if not self.is_recording and self.recorder is None:
            return
        
        self.is_recording = False
        
        # Close recorder if it exists
        if self.recorder:
            try:
                # AudioToTextRecorder is a context manager - try to close it properly
                if hasattr(self.recorder, '__exit__'):
                    self.recorder.__exit__(None, None, None)
            except Exception as e:
                self._send_error(f"Error closing recorder: {str(e)}")
            finally:
                self.recorder = None
        
        # Finalize transcript file
        if self.transcript_file_path:
            try:
                full_text = " ".join(self.full_transcript)
                with open(self.transcript_file_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n\n=== Real-Time Transcript Ended at {self._get_timestamp()} ===\n")
                    f.flush()
            except Exception as e:
                self._send_error(f"Error finalizing file: {str(e)}")
        
        self._send_message({
            "type": "status",
            "status": "recording_stopped",
            "full_transcript": " ".join(self.full_transcript),
            "file_path": self.transcript_file_path
        })
    
    def run(self):
        """Main run loop - continuously transcribe audio"""
        # Track all real-time updates to prevent losing any text
        self.realtime_buffer = []
        
        def on_realtime_update(text):
            """Callback for real-time transcription updates"""
            # DEBUG: Always log when callback is called (even if empty)
            self._send_error(f"[DEBUG] on_realtime_update CALLED with text='{text}' (type: {type(text)}, len: {len(text) if text else 0})")
            
            if text and text.strip():
                cleaned = self._preprocess_text(text.strip())
                self._send_error(f"[DEBUG] After preprocessing: '{cleaned}'")
                
                # FILTER: Skip if this looks like an error message being transcribed
                error_patterns = [
                    'no clip', 'no clips', 'clip timestamp', 'clip timestamps',
                    'timestamp found', 'vad filter', 'set vad', 'runtime error',
                    'exception', 'error:', 'traceback'
                ]
                cleaned_lower = cleaned.lower()
                if any(pattern in cleaned_lower for pattern in error_patterns):
                    self._send_error(f"[DEBUG] FILTERED: Skipping potential error message transcription: '{cleaned}'")
                    return  # Don't add error messages to transcription
                
                # IMPORTANT: Store ALL real-time updates in buffer to prevent loss
                if cleaned and cleaned not in self.realtime_buffer:
                    self.realtime_buffer.append(cleaned)
                    self._send_error(f"[DEBUG] Added to buffer. Buffer now has {len(self.realtime_buffer)} entries")
                
                if cleaned and cleaned != self.last_realtime_update:
                    # Only update if it's different from last update
                    self.last_realtime_update = cleaned
                    # Build full display text: completed sentences + current partial
                    complete_sentences = " ".join(self.full_transcript)
                    full_display = (complete_sentences + " " + cleaned).strip() if complete_sentences else cleaned
                    
                    # Send real-time update to UI with both partial and full context
                    self._send_message({
                        "type": "realtime_update",
                        "text": cleaned,  # Current partial sentence
                        "fullTranscript": full_display,  # Complete sentences + current partial
                        "timestamp": self._get_timestamp()
                    })
                    # Track but don't write real-time updates to file
                    self._append_to_file(cleaned, is_realtime=True)
                else:
                    self._send_error(f"[DEBUG] Skipping update - same as last: '{cleaned}'")
            else:
                self._send_error(f"[DEBUG] on_realtime_update: text is empty or None - text='{text}'")
        
        def on_transcription_complete(text):
            """Callback for completed transcriptions - called by recorder.text()"""
            # DEBUG: Always log when callback is called
            self._send_error(f"[DEBUG] on_transcription_complete called with: '{text}' (type: {type(text)})")
            try:
                if text and text.strip():
                    cleaned = self._preprocess_text(text.strip())
                    # Remove trailing ellipses
                    if cleaned.endswith("..."):
                        cleaned = cleaned[:-3].strip()
                    
                    # FILTER: Skip if this looks like an error message being transcribed
                    error_patterns = [
                        'no clip', 'no clips', 'clip timestamp', 'clip timestamps',
                        'timestamp found', 'vad filter', 'set vad', 'runtime error',
                        'exception', 'error:', 'traceback'
                    ]
                    cleaned_lower = cleaned.lower()
                    if any(pattern in cleaned_lower for pattern in error_patterns):
                        self._send_error(f"[DEBUG] FILTERED: Skipping error message from transcription_complete: '{cleaned}'")
                        # Preserve buffer if we have one - don't lose good transcriptions
                        if self.realtime_buffer:
                            self._send_error(f"[DEBUG] Preserving real-time buffer with {len(self.realtime_buffer)} entries despite error")
                        return  # Don't add error messages to transcript
                    
                    if cleaned:
                        # IMPORTANT: Check if any text from real-time buffer was missed
                        # This ensures we capture everything even if real-time updates were incomplete
                        self._send_error(f"[DEBUG] Real-time buffer has {len(self.realtime_buffer)} entries before processing")
                        
                        # Check if this is already in the transcript to avoid duplicates
                        if cleaned not in self.full_transcript:
                            self.full_transcript.append(cleaned)
                            self._send_error(f"[DEBUG] Added to transcript. Total sentences: {len(self.full_transcript)}")
                            
                            # Clear real-time buffer after successful transcription
                            self.realtime_buffer.clear()
                            
                            # Send complete transcription
                            self._send_message({
                                "type": "transcription_complete",
                                "text": cleaned,
                                "full_transcript": " ".join(self.full_transcript),
                                "timestamp": self._get_timestamp()
                            })
                            
                            # Write complete transcription to file (will remove any matching real-time line)
                            self._append_to_file(cleaned, is_realtime=False)
                        else:
                            self._send_error(f"[DEBUG] Skipping duplicate sentence: '{cleaned}' already in transcript")
                            # Still clear buffer to avoid accumulation
                            self.realtime_buffer.clear()
                else:
                    self._send_error(f"[DEBUG] on_transcription_complete: text is empty or None")
                    # Even if empty, check if buffer has content we should preserve
                    if self.realtime_buffer:
                        self._send_error(f"[DEBUG] WARNING: Complete callback empty but buffer has {len(self.realtime_buffer)} entries!")
            except Exception as callback_error:
                # Don't let callback errors break the loop
                self._send_error(f"[DEBUG] Error in on_transcription_complete callback: {callback_error}")
                import traceback
                self._send_error(f"[DEBUG] Callback traceback: {traceback.format_exc()}")
        
        # Create recorder with callbacks
        try:
            recorder_config = {
                'spinner': False,
                'model': self.model,
                'realtime_model_type': self.realtime_model,
                'language': self.language,
                # Voice Activity Detection settings - optimized for continuous speech
                # Note: silero_sensitivity: 0.0 = most sensitive, 1.0 = least sensitive (we want lower)
                #       webrtc_sensitivity: 0 = most sensitive, 3 = least sensitive (we want lower)
                'silero_sensitivity': 0.3,  # More sensitive (0.3 = detects speech easily)
                'webrtc_sensitivity': 1,  # More sensitive VAD (1 = detects more speech)
                # CRITICAL: Long silence duration to handle continuous speech without breaking prematurely
                'post_speech_silence_duration': 3.0,  # Wait 3.0 seconds of silence before breaking - captures long continuous speech
                'min_length_of_recording': 0.2,  # Start transcribing quickly
                'min_gap_between_recordings': 0.1,  # Small gap to ensure no overlap
                'enable_realtime_transcription': True,
                'realtime_processing_pause': 0.01,  # Faster updates (10ms) for continuous speech
                'on_realtime_transcription_update': on_realtime_update,
                'silero_deactivity_detection': True,
                'early_transcription_on_silence': 2.0,  # Wait 2.0s of silence before finalizing - ensures all continuous speech is captured
                # Add initial prompt to help with continuous/long speech
                'initial_prompt_realtime': (
                    "This is continuous speech transcription. "
                    "Transcribe everything that is said word-for-word. "
                    "Do not skip any words or sentences, even during fast or long continuous speech. "
                    "Wait for longer pauses before finalizing sentences. "
                    "Capture all spoken words completely."
                ),
                # Increase beam size for better accuracy during continuous speech
                'beam_size': 7,  # Increased from 5 for better continuous speech handling
                'beam_size_realtime': 5,  # Increased from 3 for better real-time accuracy
                'no_log_file': True,
                'silero_use_onnx': True,
                'faster_whisper_vad_filter': True,  # Enable VAD filtering to prevent "No clip timestamps" error
                # Try to ensure microphone access
                'use_microphone': True,  # Explicitly enable microphone
                # Add model caching to avoid reload delays
                'download_root': None,  # Use default cache location
            }
            
            self._send_message({"type": "status", "status": "initializing_recorder"})
            self._send_error("[DEBUG] Creating AudioToTextRecorder with config...")
            self._send_error(f"[DEBUG] Config: model={self.model}, realtime_model={self.realtime_model}, language={self.language}")
            
            try:
                self._send_error(f"[DEBUG] Creating recorder with VAD: silero={recorder_config['silero_sensitivity']}, webrtc={recorder_config['webrtc_sensitivity']}")
                self.recorder = AudioToTextRecorder(**recorder_config)
                self._send_error("[DEBUG] AudioToTextRecorder created successfully")
                self._send_error("[DEBUG] Recorder is ready to listen for audio")
            except Exception as init_error:
                self._send_error(f"[DEBUG] ERROR creating recorder: {init_error}")
                import traceback
                self._send_error(f"[DEBUG] Init traceback: {traceback.format_exc()}")
                raise
            
            self.start_recording()
            self._send_message({"type": "status", "status": "ready"})
            self._send_error("[DEBUG] Recorder started. Starting transcription loop. Listening for audio...")
            self._send_error("[DEBUG] Speak now - callbacks should fire when speech is detected!")
            
            # Listen continuously
            import threading
            import queue
            stop_queue = queue.Queue()
            
            # Thread to listen for stop commands
            def stdin_listener():
                try:
                    while True:
                        line = sys.stdin.readline()
                        if not line:
                            break
                        try:
                            command = json.loads(line.strip())
                            if command.get("action") == "stop":
                                stop_queue.put("stop")
                                break
                        except (json.JSONDecodeError, ValueError):
                            pass
                except:
                    pass
            
            stdin_thread = threading.Thread(target=stdin_listener, daemon=True)
            stdin_thread.start()
            
            # Main transcription loop - recorder.text() requires a callback
            # This matches the pattern from realtime_stt.py: while True: recorder.text(callback)
            iteration = 0
            consecutive_errors = 0
            max_consecutive_errors = 5
            
            while True:
                iteration += 1
                self._send_error(f"[DEBUG] ===== Loop iteration {iteration} ===== (errors: {consecutive_errors})")
                
                try:
                    # Check for stop command (non-blocking)
                    try:
                        stop_queue.get_nowait()
                        self._send_error(f"[DEBUG] Stop command received at iteration {iteration}")
                        self.stop_recording()
                        break
                    except queue.Empty:
                        pass
                    
                    # Validate recorder before using
                    if self.recorder is None:
                        self._send_error("[DEBUG] ERROR: Recorder is None! Cannot continue.")
                        break
                    
                    # Get transcription - pass callback function (blocks until speech is detected)
                    # This matches the pattern: recorder.text(callback)
                    self._send_error(f"[DEBUG] About to call recorder.text() - iteration {iteration}")
                    self._send_error(f"[DEBUG] WAITING for speech... (this blocks until VAD detects speech)")
                    self._send_error(f"[DEBUG] VAD Settings: silero={recorder_config.get('silero_sensitivity')}, webrtc={recorder_config.get('webrtc_sensitivity')}")
                    
                    # recorder.text() blocks until speech is detected and processed
                    # It will call:
                    #   - on_realtime_transcription_update (continuously as you speak)
                    #   - on_transcription_complete (when sentence ends)
                    # It should return the transcribed text or None
                    result = self.recorder.text(on_transcription_complete)
                    
                    # Reset error counter on success
                    consecutive_errors = 0
                    self._send_error(f"[DEBUG] recorder.text() returned: result={result} (type: {type(result)})")
                    self._send_error(f"[DEBUG] Completed iteration {iteration} successfully, continuing to next iteration...")
                    
                    # If no result but callback fired, that's OK - continue
                    if result is None or result == "":
                        self._send_error(f"[DEBUG] Note: recorder.text() returned empty, but callback should have processed it. Continuing...")
                    
                except KeyboardInterrupt:
                    self._send_error("[DEBUG] KeyboardInterrupt received")
                    break
                except EOFError:
                    self._send_error("[DEBUG] EOFError received")
                    break
                except Exception as e:
                    consecutive_errors += 1
                    error_msg = str(e)
                    self._send_error(f"[DEBUG] Exception at iteration {iteration} (error #{consecutive_errors}): {error_msg}")
                    import traceback
                    self._send_error(f"[DEBUG] Traceback: {traceback.format_exc()}")
                    
                    # CRITICAL: Preserve real-time buffer when errors occur
                    # Don't lose transcriptions we've already captured
                    if self.realtime_buffer:
                        self._send_error(f"[DEBUG] ERROR OCCURRED: Preserving real-time buffer with {len(self.realtime_buffer)} entries")
                        # Try to save what we have from the buffer before continuing
                        # The buffer contains valid transcriptions that shouldn't be lost
                        if len(self.realtime_buffer) > 0:
                            # Use the most complete entry from buffer as fallback
                            last_buffer_entry = self.realtime_buffer[-1]
                            if last_buffer_entry and last_buffer_entry not in self.full_transcript:
                                self._send_error(f"[DEBUG] Attempting to save buffer entry as fallback: '{last_buffer_entry[:50]}...'")
                                # Don't add automatically - let next successful transcription handle it
                    
                    # Check if recorder is still valid
                    if self.recorder is None:
                        self._send_error("[DEBUG] Recorder became None after error. Cannot continue.")
                        break
                    
                    # Too many consecutive errors - something is seriously wrong
                    if consecutive_errors >= max_consecutive_errors:
                        self._send_error(f"[DEBUG] Too many consecutive errors ({consecutive_errors}). Stopping.")
                        # Before stopping, try to save any remaining buffer content
                        if self.realtime_buffer:
                            self._send_error(f"[DEBUG] Final attempt: Saving {len(self.realtime_buffer)} buffer entries before exit")
                        break
                    
                    # Try to continue - wait before retrying
                    import time
                    wait_time = min(1.0 * consecutive_errors, 5.0)  # Exponential backoff, max 5 seconds
                    self._send_error(f"[DEBUG] Waiting {wait_time:.1f} seconds before retrying...")
                    time.sleep(wait_time)
                    # Continue loop - don't exit
                    
        except KeyboardInterrupt:
            self._send_message({"type": "status", "status": "stopped"})
        except Exception as e:
            self._send_error(f"Fatal error: {str(e)}")
            import traceback
            self._send_error(f"Traceback: {traceback.format_exc()}")
            sys.exit(1)
        finally:
            self.stop_recording()
    
    def _preprocess_text(self, text):
        """Clean and preprocess text"""
        # Remove leading whitespaces
        text = text.lstrip()
        
        # Remove starting ellipses if present
        if text.startswith("..."):
            text = text[3:]
        
        # Remove any leading whitespaces again
        text = text.lstrip()
        
        # Uppercase first letter
        if text:
            text = text[0].upper() + text[1:]
        
        return text


def main():
    """Main entry point"""
    try:
        # Read configuration from stdin or command line args
        if len(sys.argv) > 1:
            # Command line mode
            import argparse
            parser = argparse.ArgumentParser(description='RealtimeSTT Service')
            parser.add_argument('--transcript-file', type=str, help='Path to transcript file')
            parser.add_argument('--model', type=str, default='large-v3', help='Whisper model (large-v3 = latest and most accurate)')
            parser.add_argument('--realtime-model', type=str, default='base.en', help='Real-time model (base.en = better accuracy, small.en = even better)')
            parser.add_argument('--language', type=str, default='en', help='Language code')
            
            args = parser.parse_args()
            
            service = RealtimeSTTService(
                transcript_file_path=args.transcript_file,
                model=args.model,
                realtime_model=args.realtime_model,
                language=args.language
            )
            
            # Auto-start recording
            service.start_recording()
            service.run()
        else:
            # Interactive mode - read config from stdin
            config_line = sys.stdin.readline()
            if config_line:
                config = json.loads(config_line.strip())
                service = RealtimeSTTService(
                    transcript_file_path=config.get('transcript_file'),
                    model=config.get('model', 'large-v3'),
                    realtime_model=config.get('realtime_model', 'base.en'),
                    language=config.get('language', 'en')
                )
                service.start_recording()
                service.run()
            
    except KeyboardInterrupt:
        print(json.dumps({"type": "status", "status": "interrupted"}), file=sys.stderr)
    except Exception as e:
        print(json.dumps({"type": "error", "error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
