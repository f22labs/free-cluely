#!/usr/bin/env python3
"""
RealtimeSTT Service for Electron Integration
This service provides real-time transcription using RealtimeSTT library.
Communicates via JSON over stdin/stdout for IPC with Electron.
"""

import sys
import json
import os
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Any

try:
    from RealtimeSTT import AudioToTextRecorder
except ImportError:
    _ts = datetime.now().isoformat()
    print(
        json.dumps({
            "type": "error",
            "error": f"[{_ts}] RealtimeSTT not installed. Run: pip install RealtimeSTT",
            "timestamp": _ts,
        }),
        file=sys.stderr,
    )
    sys.exit(1)

class RealtimeSTTService:
    def __init__(self, transcript_file_path=None, model='large-v3', realtime_model='base.en', language='en'):
        """Initialize the RealtimeSTT recorder"""
        self.transcript_file_path = transcript_file_path
        self.transcript_file_header = None
        self.session_start_iso = self._get_timestamp()
        self.full_transcript = []
        self.is_recording = False
        self.last_written_text = ""  # Track last written text to avoid duplicates
        self.last_realtime_update = ""  # Track last real-time update
        self.realtime_buffer = []  # Buffer to store all real-time updates as backup
        self.current_iteration_start = None
        self.current_iteration_index = 0
        self.partial_logged = False
        self.last_audio_detected = None
        self.audio_watchdog_stop: threading.Event | None = None
        self.audio_watchdog_thread: threading.Thread | None = None
        self.audio_watchdog_last_warning = None
        self.current_iteration_wallclock_iso = None
        self.current_iteration_start_epoch_ms = None
        self.current_iteration_monotonic_ms = None
        self.first_partial_latency_ms = None
        self.partial_update_count = 0
        self.debug_enabled = os.getenv("RTSTT_DEBUG", "").lower() in ("1", "true", "yes")

        # Suggestion timeout management
        self.suggestion_timeout_seconds = float(os.getenv("RTSTT_SUGGESTION_TIMEOUT_SECONDS", "3.0"))
        self._suggestion_timer: threading.Timer | None = None
        self._suggestion_timer_iteration: int | None = None
        self._suggestion_timer_triggered = False
        self._suggestion_timer_payload: dict[str, Any] | None = None
        self._suggestion_timer_start_monotonic: float | None = None
        self._suggestion_timer_start_epoch_ms: int | None = None
        self._suggestion_timer_lock = threading.Lock()
        self._suggestion_timeout_counts: dict[int, int] = {}
        
        # Create transcript file if path provided
        if transcript_file_path:
            os.makedirs(os.path.dirname(transcript_file_path), exist_ok=True)
            header = f"=== Real-Time Transcript Started at {self.session_start_iso} ===\n\n"
            with open(transcript_file_path, 'w', encoding='utf-8') as f:
                f.write(header)
            self.transcript_file_header = header
        
        # Store config for later use in run()
        self.model = model
        self.realtime_model = realtime_model
        self.language = language
        
        # Don't create recorder here - will be created in run() with callbacks
        self.recorder = None
        self._send_message({"type": "status", "status": "initialized", "timestamp": self._get_timestamp()})
    
    def _get_timestamp(self):
        return datetime.now().isoformat()
    
    def _get_epoch_ms(self):
        return int(time.time() * 1000)

    def _send_message(self, message):
        """Send JSON message to stdout"""
        try:
            print(json.dumps(message), flush=True)
        except Exception as e:
            self._send_error(f"Error sending message: {str(e)}")
    
    def _send_error(self, error_msg):
        """Send error message"""
        timestamp = self._get_timestamp()
        error_text = error_msg if isinstance(error_msg, str) else str(error_msg)
        print(json.dumps({"type": "error", "error": error_text, "timestamp": timestamp}), file=sys.stderr, flush=True)

    def _log(self, level, message):
        if level == "debug" and not self.debug_enabled:
            return
        timestamp = self._get_timestamp()
        text = message if isinstance(message, str) else str(message)
        print(
            json.dumps({"type": "log", "level": level, "message": text, "timestamp": timestamp}),
            file=sys.stderr,
            flush=True,
        )

    def _debug(self, message):
        self._log("debug", message)

    def _info(self, message):
        self._log("info", message)

    def _warn(self, message):
        self._log("warn", message)
    
    def _cancel_suggestion_timeout(self, iteration: int | None = None):
        """Cancel any pending suggestion timeout for the current iteration."""
        with self._suggestion_timer_lock:
            if iteration is not None and self._suggestion_timer_iteration != iteration:
                return
            if self._suggestion_timer:
                self._suggestion_timer.cancel()
            self._suggestion_timer = None
            self._suggestion_timer_iteration = None
            self._suggestion_timer_triggered = False
            self._suggestion_timer_payload = None
            self._suggestion_timer_start_monotonic = None
            self._suggestion_timer_start_epoch_ms = None
            if iteration is not None:
                self._suggestion_timeout_counts.pop(iteration, None)

    def _schedule_suggestion_timeout(self, partial_text: str, full_transcript: str):
        """
        Schedule a fallback suggestion trigger if the final transcription
        does not arrive within the configured timeout window.
        """
        if not partial_text or self.suggestion_timeout_seconds <= 0:
            return

        iteration = self.current_iteration_index
        if iteration is None:
            return

        now_monotonic = time.monotonic()
        now_epoch_ms = self._get_epoch_ms()

        with self._suggestion_timer_lock:
            # Always keep the latest partial transcript in payload
            payload_metrics = {
                "iteration": iteration,
                "partial_index": self.partial_update_count,
                "first_partial_latency_ms": self.first_partial_latency_ms,
                "python_iteration_started_at": self.current_iteration_wallclock_iso,
                "python_iteration_started_epoch_ms": self.current_iteration_start_epoch_ms,
            }
            self._suggestion_timer_payload = {
                "text": partial_text,
                "full_transcript": full_transcript,
                "metrics": payload_metrics,
            }

            if self._suggestion_timer is None or self._suggestion_timer_iteration != iteration:
                # Cancel any pending timer for a previous iteration
                if self._suggestion_timer and self._suggestion_timer_iteration != iteration:
                    self._suggestion_timer.cancel()

                self._suggestion_timer_iteration = iteration
                self._suggestion_timer_triggered = False
                self._suggestion_timer_start_monotonic = self.current_iteration_start or now_monotonic
                self._suggestion_timer_start_epoch_ms = self.current_iteration_start_epoch_ms or now_epoch_ms

                self._suggestion_timer = threading.Timer(
                    self.suggestion_timeout_seconds,
                    self._emit_suggestion_timeout
                )
                self._suggestion_timer.daemon = True
                self._suggestion_timer.start()

    def _emit_suggestion_timeout(self):
        """Emit a partial transcription event when the final transcript is delayed."""
        with self._suggestion_timer_lock:
            if self._suggestion_timer_triggered or not self._suggestion_timer_payload:
                return

            payload = dict(self._suggestion_timer_payload)
            iteration = self._suggestion_timer_iteration
            start_monotonic = self._suggestion_timer_start_monotonic
            start_epoch_ms = self._suggestion_timer_start_epoch_ms

            self._suggestion_timer_triggered = True
            self._suggestion_timer = None

        if iteration is None:
            return

        elapsed_ms = None
        if start_monotonic is not None:
            elapsed_ms = max(0.0, (time.monotonic() - start_monotonic) * 1000.0)
        else:
            elapsed_ms = self.suggestion_timeout_seconds * 1000.0

        with self._suggestion_timer_lock:
            timeout_sequence = self._suggestion_timeout_counts.get(iteration, 0) + 1
            self._suggestion_timeout_counts[iteration] = timeout_sequence

        metrics = payload.get("metrics", {}) or {}
        metrics.update({
            "iteration": iteration,
            "timeout_elapsed_ms": elapsed_ms,
            "timeout_seconds": self.suggestion_timeout_seconds,
            "timeout_sequence": timeout_sequence,
            "python_iteration_started_at": metrics.get("python_iteration_started_at"),
            "python_iteration_started_epoch_ms": metrics.get("python_iteration_started_epoch_ms"),
            "partial_update_count": self.partial_update_count,
        })

        message = {
            "type": "transcription_timeout",
            "text": payload.get("text", ""),
            "full_transcript": payload.get("full_transcript", payload.get("text", "")),
            "timestamp": self._get_timestamp(),
            "metrics": metrics,
        }

        self._send_message(message)

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
                    self._debug(f"Error reading file for duplicate removal: {str(e)}")
                
                # Append the new complete transcription
                with open(self.transcript_file_path, 'a', encoding='utf-8') as f:
                    f.write(text_stripped + "\n")
                    f.flush()
                
                self.last_written_text = text_stripped
                self.last_realtime_update = ""  # Reset after writing complete
            except Exception as e:
                self._send_error(f"Error writing to file: {str(e)}")

    def _write_full_transcript_to_file(self, include_footer=False):
        """Write the entire accumulated transcript to disk."""
        if not self.transcript_file_path:
            return

        try:
            os.makedirs(os.path.dirname(self.transcript_file_path), exist_ok=True)

            lines = []
            header = self.transcript_file_header or f"=== Real-Time Transcript Started at {self.session_start_iso} ===\n\n"
            lines.append(header if header.endswith("\n") else header + "\n")

            for entry in [t.strip() for t in self.full_transcript if t.strip()]:
                lines.append(entry + "\n")

            if include_footer:
                lines.append("\n")
                lines.append(f"=== Real-Time Transcript Ended at {self._get_timestamp()} ===\n")

            with open(self.transcript_file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
        except Exception as e:
            self._debug(f"Error writing full transcript to file: {e}")
    
    def start_recording(self):
        """Start recording and transcription"""
        if self.is_recording:
            return
        
        self.is_recording = True
        self._send_message({"type": "status", "status": "recording_started", "timestamp": self._get_timestamp()})
    
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
                self._write_full_transcript_to_file(include_footer=True)
            except Exception as e:
                self._send_error(f"Error finalizing file: {str(e)}")
        
        self._send_message({
            "type": "status",
            "status": "recording_stopped",
            "full_transcript": "\n".join(self.full_transcript),
            "file_path": self.transcript_file_path,
            "timestamp": self._get_timestamp()
        })
    
    def run(self):
        """Main run loop - continuously transcribe audio"""
        # Track all real-time updates to prevent losing any text
        self.realtime_buffer = []
        self.last_audio_detected = time.monotonic()
        
        def on_realtime_update(text):
            """Callback for real-time transcription updates"""
            self._info(f"[Transcription Update] iteration {self.current_iteration_index}: {text}")
            now_monotonic = time.monotonic()
            self.last_audio_detected = now_monotonic

            if not text:
                return

            cleaned = self._preprocess_text(text.strip())
            if not cleaned:
                return

            error_patterns = [
                'no clip', 'no clips', 'clip timestamp', 'clip timestamps',
                'timestamp found', 'vad filter', 'set vad', 'runtime error',
                'exception', 'error:', 'traceback'
            ]
            cleaned_lower = cleaned.lower()
            if any(pattern in cleaned_lower for pattern in error_patterns):
                return

            if cleaned not in self.realtime_buffer:
                self.realtime_buffer.append(cleaned)

            if cleaned == self.last_realtime_update:
                return

            self.last_realtime_update = cleaned
            self.partial_update_count += 1

            latency_ms = None
            if self.current_iteration_start is not None:
                latency_ms = round((now_monotonic - self.current_iteration_start) * 1000, 2)
                if not self.partial_logged:
                    self.first_partial_latency_ms = latency_ms
                self.partial_logged = True

            complete_sentences = "\n".join(self.full_transcript)
            full_display = f"{complete_sentences}\n{cleaned}".strip() if complete_sentences else cleaned

            emit_timestamp_iso = self._get_timestamp()
            emit_epoch_ms = self._get_epoch_ms()

            self._send_message({
                "type": "realtime_update",
                "text": cleaned,
                "fullTranscript": full_display,
                "timestamp": emit_timestamp_iso,
                "metrics": {
                    "iteration": self.current_iteration_index,
                    "partial_index": self.partial_update_count,
                    "latency_ms": latency_ms,
                    "first_partial_latency_ms": self.first_partial_latency_ms,
                    "python_iteration_started_at": self.current_iteration_wallclock_iso,
                    "python_iteration_started_epoch_ms": self.current_iteration_start_epoch_ms,
                    "python_emit_timestamp": emit_timestamp_iso,
                    "python_emit_epoch_ms": emit_epoch_ms
                }
            })
            self._append_to_file(cleaned, is_realtime=True)

            # Start a suggestion timeout if we have not received a final transcription yet.
            self._schedule_suggestion_timeout(cleaned, full_display)
        
        def on_transcription_complete(text):
            """Callback for completed transcriptions - called by recorder.text()"""
            completion_monotonic = time.monotonic()
            self.last_audio_detected = completion_monotonic
            total_latency_ms = None
            if self.current_iteration_start is not None:
                total_latency_ms = round((completion_monotonic - self.current_iteration_start) * 1000, 2)
            # Final transcription arrived, cancel any pending timeout for this iteration
            self._cancel_suggestion_timeout(self.current_iteration_index)
            if self.debug_enabled and text:
                self._debug(f"Transcription complete (iteration {self.current_iteration_index}): {text}")
            try:
                if text and text.strip():
                    cleaned = self._preprocess_text(text.strip())
                    # Remove trailing ellipses
                    if cleaned.endswith("..."):
                        cleaned = cleaned[:-3].strip()
                    
                    self._info(f"[Transcription Complete] iteration {self.current_iteration_index}: {cleaned}")
                    
                    # FILTER: Skip if this looks like an error message being transcribed
                    error_patterns = [
                        'no clip', 'no clips', 'clip timestamp', 'clip timestamps',
                        'timestamp found', 'vad filter', 'set vad', 'runtime error',
                        'exception', 'error:', 'traceback'
                    ]
                    cleaned_lower = cleaned.lower()
                    if any(pattern in cleaned_lower for pattern in error_patterns):
                        return  # Don't add error messages to transcript
                    
                    if cleaned:
                        if self.realtime_buffer:
                            # Get the most complete entry from buffer (usually the last one)
                            buffer_text = self.realtime_buffer[-1] if self.realtime_buffer else ""
                            
                            # If buffer text is significantly longer or more complete, it might have more words
                            # Use the longer/more detailed version, prioritizing buffer during fast speech
                            if buffer_text and len(buffer_text) > len(cleaned) * 1.2:  # Buffer is 20%+ longer
                                # Prefer buffer if it's substantially longer (likely more complete for fast speech)
                                if buffer_text.lower() not in cleaned.lower() and cleaned.lower() not in buffer_text.lower():
                                    # They're different - merge them intelligently
                                    # Use buffer if it contains the final text plus more
                                    if cleaned.lower() in buffer_text.lower():
                                        cleaned = buffer_text  # Use the longer buffer version
                        
                        # Check if this is already in the transcript to avoid duplicates
                        if cleaned not in self.full_transcript:
                            self.full_transcript.append(cleaned)
                            
                            buffer_entry_count = len(self.realtime_buffer)
                            completion_iso = self._get_timestamp()
                            completion_epoch_ms = self._get_epoch_ms()
                            metrics_payload = {
                                "iteration": self.current_iteration_index,
                                "transcription_latency_ms": total_latency_ms,
                                "first_partial_latency_ms": self.first_partial_latency_ms,
                                "partial_update_count": self.partial_update_count,
                                "python_iteration_started_at": self.current_iteration_wallclock_iso,
                                "python_iteration_started_epoch_ms": self.current_iteration_start_epoch_ms,
                                "python_completion_timestamp": completion_iso,
                                "python_completion_epoch_ms": completion_epoch_ms,
                                "python_emit_timestamp": completion_iso,
                                "python_emit_epoch_ms": completion_epoch_ms,
                                "realtime_buffer_entries": buffer_entry_count,
                                "fallback_used": False
                            }
                            
                            # Clear real-time buffer AFTER successful transcription and merging
                            self.realtime_buffer.clear()
                            
                            # Send complete transcription
                            self._send_message({
                                "type": "transcription_complete",
                                "text": cleaned,
                                "full_transcript": "\n".join(self.full_transcript),
                                "timestamp": completion_iso,
                                "metrics": metrics_payload
                            })
                            
                            # Persist full transcript to disk
                            self._write_full_transcript_to_file()
                        else:
                            self._debug(f"Skipping duplicate sentence: '{cleaned}' already in transcript")
                            self.realtime_buffer.clear()
                else:
                    self._debug("on_transcription_complete received empty text")
                    # CRITICAL: If final transcription is empty but buffer has content, use buffer!
                    # This handles cases where fast speech causes final transcription to fail
                    if self.realtime_buffer:
                        self._debug(f"Complete callback empty but buffer has {len(self.realtime_buffer)} entries")
                        # Use the most complete buffer entry as fallback
                        buffer_text = self.realtime_buffer[-1] if self.realtime_buffer else ""
                        if buffer_text and buffer_text.strip():
                            cleaned = self._preprocess_text(buffer_text.strip())
                            if cleaned and cleaned not in self.full_transcript:
                                self._debug(f"Using buffer text '{cleaned[:50]}...' as fallback")
                                self.full_transcript.append(cleaned)
                                
                                buffer_entry_count = len(self.realtime_buffer)
                                completion_iso = self._get_timestamp()
                                completion_epoch_ms = self._get_epoch_ms()
                                metrics_payload = {
                                    "iteration": self.current_iteration_index,
                                    "transcription_latency_ms": total_latency_ms,
                                    "first_partial_latency_ms": self.first_partial_latency_ms,
                                    "partial_update_count": self.partial_update_count,
                                    "python_iteration_started_at": self.current_iteration_wallclock_iso,
                                    "python_iteration_started_epoch_ms": self.current_iteration_start_epoch_ms,
                                    "python_completion_timestamp": completion_iso,
                                    "python_completion_epoch_ms": completion_epoch_ms,
                                    "python_emit_timestamp": completion_iso,
                                    "python_emit_epoch_ms": completion_epoch_ms,
                                    "realtime_buffer_entries": buffer_entry_count,
                                    "fallback_used": True
                                }
                                
                                self._send_message({
                                    "type": "transcription_complete",
                                    "text": cleaned,
                                    "full_transcript": "\n".join(self.full_transcript),
                                    "timestamp": completion_iso,
                                    "metrics": metrics_payload
                                })
                                self._write_full_transcript_to_file()
                            self.realtime_buffer.clear()
            except Exception as callback_error:
                # Don't let callback errors break the loop
                self._debug(f"Error in on_transcription_complete callback: {callback_error}")
                import traceback
                self._debug(f"Callback traceback: {traceback.format_exc()}")
        
        # Platform detection and M1-specific diagnostics
        import platform
        system_info = platform.system()
        machine = platform.machine()
        self._info(f"[PLATFORM] System: {system_info}, Architecture: {machine}")
        
        # Check for Apple Silicon (M1/M2/M3/M4)
        is_apple_silicon = machine == 'arm64' and system_info == 'Darwin'
        if is_apple_silicon:
            self._info("[PLATFORM] Detected Apple Silicon (M1/M2/M3/M4) - checking compatibility...")
            
            # Test microphone access before creating recorder
            try:
                import pyaudio
                p_test = pyaudio.PyAudio()
                try:
                    default_input = p_test.get_default_input_device_info()
                    self._info(f"[PLATFORM] Microphone detected: {default_input['name']} (index={default_input['index']}, sample_rate={default_input['defaultSampleRate']} Hz)")
                    
                    # Try to open a test stream
                    test_stream = p_test.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=1024
                    )
                    self._info("[PLATFORM] Microphone stream test passed on Apple Silicon")
                    test_stream.stop_stream()
                    test_stream.close()
                except Exception as mic_error:
                    self._warn(f"[PLATFORM] Microphone access test failed on Apple Silicon: {mic_error}")
                    self._warn("[PLATFORM] Possible causes: permissions not granted, PyAudio mismatch, or PortAudio issue")
                finally:
                    p_test.terminate()
            except ImportError:
                self._debug("[PLATFORM] PyAudio not available for microphone test")
            except Exception as e:
                self._warn(f"[PLATFORM] Error testing microphone: {e}")
        
        # Create recorder with callbacks
        try:
            recorder_config = {
                'spinner': False,
                'model': self.model,
                'realtime_model_type': self.realtime_model,
                'language': self.language,
                # Voice Activity Detection settings - OPTIMIZED FOR FAST SPEECH & LONG SILENCE
                # Note: silero_sensitivity: 0.0 = most sensitive, 1.0 = least sensitive (we want MUCH lower for fast speech)
                #       webrtc_sensitivity: 0 = most sensitive, 3 = least sensitive (we want 0 for maximum sensitivity)
                'silero_sensitivity': 0.15,  # VERY sensitive (0.15 = catches fast speech and speech after silence)
                'webrtc_sensitivity': 0,  # MAXIMUM sensitivity (0 = catches every speech event, even after long silence)
                # CRITICAL: Balanced silence duration - short enough to detect fast speech, long enough for continuous speech
                'post_speech_silence_duration': 2.5,  # 2.5s - faster response than 3.0s, but still captures long speech
                'min_length_of_recording': 0.3,  # 0.3s - slightly longer for better quality, prevents premature cuts on fast speech
                'min_gap_between_recordings': 0.05,  # 0.05s - smaller gap for faster detection, ensures no missed words
                'enable_realtime_transcription': True,
                'realtime_processing_pause': 0.01,  # Faster updates (10ms) for continuous speech
                'on_realtime_transcription_update': on_realtime_update,
                'silero_deactivity_detection': True,  # Enable deactivity detection for better silence handling
                'early_transcription_on_silence': 1.8,  # 1.8s - faster finalization, prevents waiting too long (good for fast speech)
                # Additional VAD settings for better detection
                'silero_use_onnx': True,  # Use ONNX for faster VAD processing
                # Optimized beam size for better accuracy during fast and continuous speech
                'beam_size': 10,  # Higher beam size for final transcription = better accuracy
                'beam_size_realtime': 7,  # Higher real-time beam = better accuracy for fast speech (7-8 is optimal balance)
                # Note: compression_ratio_threshold, log_prob_threshold, and no_speech_threshold are not 
                # direct AudioToTextRecorder parameters - they're lower-level Whisper parameters
                # that are handled internally by the model configuration
                'no_log_file': False,
                'faster_whisper_vad_filter': True,  # Enable VAD filtering to prevent "No clip timestamps" error
                # Try to ensure microphone access
                'use_microphone': True,  # Explicitly enable microphone
                # Add model caching to avoid reload delays
                'download_root': None,  # Use default cache location
            }
            
            self._send_message({"type": "status", "status": "initializing_recorder", "timestamp": self._get_timestamp()})
            self._debug("Creating AudioToTextRecorder with config")
            self._debug(f"Config: model={self.model}, realtime_model={self.realtime_model}, language={self.language}")
            
            try:
                self._debug(f"Creating recorder with VAD: silero={recorder_config['silero_sensitivity']}, webrtc={recorder_config['webrtc_sensitivity']}")
                
                if is_apple_silicon:
                    self._debug("Creating AudioToTextRecorder on Apple Silicon (first load may take longer)")
                
                self.recorder = AudioToTextRecorder(**recorder_config)
                self._debug("AudioToTextRecorder created successfully and ready to listen")
                
                if is_apple_silicon:
                    try:
                        import pyaudio
                        p_verify = pyaudio.PyAudio()
                        default_input = p_verify.get_default_input_device_info()
                        self._info(f"[PLATFORM] Recorder active on microphone: {default_input['name']} (index={default_input['index']})")
                        p_verify.terminate()
                    except Exception as verify_error:
                        self._warn(f"[PLATFORM] Could not verify microphone after recorder creation: {verify_error}")
                
            except Exception as init_error:
                self._send_error(f"Error creating recorder: {init_error}")
                if is_apple_silicon:
                    self._warn("Recorder creation failed on Apple Silicon")
                    self._info("[PLATFORM] Common fixes for M1/M2/M3:")
                    self._info("  1. Install PortAudio: brew install portaudio")
                    self._info("  2. Reinstall PyAudio: pip install --force-reinstall pyaudio")
                    self._info("  3. Check microphone permissions in System Settings")
                    self._info("  4. Try running from Terminal (not IDE) to get permission prompt")
                import traceback
                self._debug(f"Init traceback: {traceback.format_exc()}")
                raise
            
            self.start_recording()
            self._send_message({"type": "status", "status": "ready", "timestamp": self._get_timestamp()})
            self._debug("Recorder started; transcription loop active")
            self._debug("Callbacks will fire when speech is detected")
            
            # Listen continuously
            import threading
            import queue
            stop_queue = queue.Queue()
            
            # Thread to listen for stop commands
            def stdin_listener():
                try:
                    while True:
                        self._debug("Waiting for commands on stdin")
                        line = sys.stdin.readline()
                        if self.debug_enabled:
                            self._debug(f"stdin line: {line.rstrip()}")
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
            self._debug("stdin listener thread started")
            
            self.audio_watchdog_stop = threading.Event()
            self.audio_watchdog_last_warning = None

            def audio_watchdog():
                self._debug("Audio watchdog thread started")
                while not self.audio_watchdog_stop.is_set():
                    time.sleep(3)
                    if not self.is_recording:
                        continue
                    if self.last_audio_detected is None:
                        continue
                    gap = time.monotonic() - self.last_audio_detected
                    if gap > 5:
                        if (
                            self.audio_watchdog_last_warning is None
                            or (time.monotonic() - self.audio_watchdog_last_warning) > 5
                        ):
                            self.audio_watchdog_last_warning = time.monotonic()
                            self._info(f"[AUDIO WATCHDOG] No audio detected for {gap:.2f} seconds. Waiting for speech input...")

            self.audio_watchdog_thread = threading.Thread(target=audio_watchdog, daemon=True)
            self.audio_watchdog_thread.start()
            
            # Main transcription loop - recorder.text() requires a callback
            # This matches the pattern from realtime_stt.py: while True: recorder.text(callback)
            iteration = 0
            consecutive_errors = 0
            max_consecutive_errors = 5
            
            self._debug("Starting main transcription loop")
            while True:
                iteration += 1
                self._debug(f"Entering loop iteration {iteration} (consecutive errors: {consecutive_errors})")
                
                try:
                    # Check for stop command (non-blocking)
                    try:
                        stop_queue.get_nowait()
                        self._debug(f"Stop command received at iteration {iteration}")
                        self.stop_recording()
                        break
                    except queue.Empty:
                        pass
                    
                    # Validate recorder before using
                    if self.recorder is None:
                        self._send_error("Recorder is None. Stopping transcription loop.")
                        break
                    
                    # Reset iteration tracking
                    now_monotonic = time.monotonic()
                    self.current_iteration_index = iteration
                    self.current_iteration_start = now_monotonic
                    self.current_iteration_monotonic_ms = now_monotonic * 1000
                    self.current_iteration_wallclock_iso = self._get_timestamp()
                    self.current_iteration_start_epoch_ms = self._get_epoch_ms()
                    self.first_partial_latency_ms = None
                    self.partial_update_count = 0
                    self.partial_logged = False
                    if self.last_audio_detected is None:
                        self.last_audio_detected = now_monotonic

                    # Get transcription - pass callback function (blocks until speech is detected)
                    # This matches the pattern: recorder.text(callback)
                    self._debug(f"Starting recorder.text() for iteration {iteration}")
                    self._debug(f"VAD Settings: silero={recorder_config.get('silero_sensitivity')}, webrtc={recorder_config.get('webrtc_sensitivity')}")
                    
                    # recorder.text() blocks until speech is detected and processed
                    # It will call:
                    #   - on_realtime_transcription_update (continuously as you speak)
                    #   - on_transcription_complete (when sentence ends)
                    # It should return the transcribed text or None
                    self._debug(f"Calling recorder.text() - iteration {iteration}")
                    
                    if is_apple_silicon:
                        self._debug("Recorder running on Apple Silicon; monitor callbacks for microphone access issues")
                    start_block = time.monotonic()
                    # Call recorder.text() - this blocks until speech is detected
                    result = self.recorder.text(on_transcription_complete)
                    block_duration = time.monotonic() - start_block
                    self._info(
                        f"[BLOCK] recorder.text() iteration {iteration} "
                        f"duration={block_duration:.2f}s "
                        f"partial_logged={self.partial_logged}"
                    )
                    
                    self._debug(f"recorder.text() completed for iteration {iteration}")

                    # Ensure suggestion timeout is cleared once recorder.text completes.
                    self._cancel_suggestion_timeout(iteration)
                    
                    if (result is None or result == "") and not self.partial_logged:
                        self._warn(f"[AUDIO] recorder.text() returned empty without partial updates (iteration {self.current_iteration_index})")
                    
                    self.current_iteration_start = None
                    self.current_iteration_wallclock_iso = None
                    self.current_iteration_start_epoch_ms = None
                    self.current_iteration_monotonic_ms = None
                    self.first_partial_latency_ms = None
                    self.partial_update_count = 0
                    self.partial_logged = False
                    
                    # Reset error counter on success
                    consecutive_errors = 0
                    self._debug(f"Iteration {iteration} completed successfully (result type: {type(result)})")
                    
                    # If no result but callback fired, that's OK - continue
                    if result is None or result == "":
                        self._debug("recorder.text() returned empty, callback already handled transcription")
                    
                except KeyboardInterrupt:
                    self._debug("KeyboardInterrupt received")
                    break
                except EOFError:
                    self._debug("EOFError received")
                    break
                except Exception as e:
                    consecutive_errors += 1
                    error_msg = str(e)
                    self._debug(f"Exception at iteration {iteration} (error #{consecutive_errors}): {error_msg}")
                    import traceback
                    self._debug(f"Traceback: {traceback.format_exc()}")
                    
                    # CRITICAL: Preserve real-time buffer when errors occur
                    # Don't lose transcriptions we've already captured
                    if self.realtime_buffer:
                        self._debug(f"Preserving real-time buffer with {len(self.realtime_buffer)} entries")
                        # Try to save what we have from the buffer before continuing
                        # The buffer contains valid transcriptions that shouldn't be lost
                        if len(self.realtime_buffer) > 0:
                            # Use the most complete entry from buffer as fallback
                            last_buffer_entry = self.realtime_buffer[-1]
                            if last_buffer_entry and last_buffer_entry not in self.full_transcript:
                                self._debug(f"Attempting to save buffer entry as fallback: '{last_buffer_entry[:50]}...'")
                                # Don't add automatically - let next successful transcription handle it
                    
                    # Check if recorder is still valid
                    if self.recorder is None:
                        self._debug("Recorder became None after error. Cannot continue.")
                        break
                    
                    # Too many consecutive errors - something is seriously wrong
                    if consecutive_errors >= max_consecutive_errors:
                        self._debug(f"Too many consecutive errors ({consecutive_errors}). Stopping.")
                        # Before stopping, try to save any remaining buffer content
                        if self.realtime_buffer:
                            self._debug(f"Final attempt: Saving {len(self.realtime_buffer)} buffer entries before exit")
                        break
                    
                    # Try to continue - wait before retrying
                    wait_time = min(1.0 * consecutive_errors, 5.0)  # Exponential backoff, max 5 seconds
                    self._debug(f"Waiting {wait_time:.1f} seconds before retrying...")
                    time.sleep(wait_time)
                    # Continue loop - don't exit
                    
        except KeyboardInterrupt:
            self._send_message({"type": "status", "status": "stopped", "timestamp": self._get_timestamp()})
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
        _ts = datetime.now().isoformat()
        print(json.dumps({"type": "status", "status": "interrupted", "timestamp": _ts}), file=sys.stderr)
    except Exception as e:
        _ts = datetime.now().isoformat()
        print(json.dumps({"type": "error", "error": f"[{_ts}] {e}", "timestamp": _ts}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
