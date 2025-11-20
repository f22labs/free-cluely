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
        self._preserve_recorder = False  # Flag to preserve recorder in pre-init mode
        self._shared_stop_queue = None  # Shared stop queue for pre-init mode
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
        # Also print to stdout for critical debug messages
        if "[CALLBACK]" in text or "[TRANSCRIPTION_DEBUG]" in text:
            print(f"[PYTHON_STDOUT] {text}", flush=True)
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
        self._debug(f"[start_recording()] Called - is_recording: {self.is_recording}, recorder exists: {self.recorder is not None}, _preserve_recorder: {self._preserve_recorder}")
        if self.is_recording:
            self._debug("[start_recording()] Already recording, returning early")
            return
        
        self.is_recording = True
        self._debug("[start_recording()] Set is_recording = True")
        
        # CRITICAL: Start the recorder to enable voice activity detection and recording
        if self.recorder is not None:
            try:
                # Start the recorder - this enables voice activity detection
                # The recorder will auto-start recording when voice is detected
                self.recorder.start()
                self._debug("[start_recording()] Called recorder.start() - voice activity detection enabled")
            except Exception as e:
                self._warn(f"[start_recording()] Error calling recorder.start(): {e}")
        
        # When reusing a pre-initialized recorder, don't send "recording_started" 
        # because the recorder is already ready. The "recorder_ready" message from run() 
        # will indicate the recorder is ready, avoiding the "preparing recorder" status.
        if self._preserve_recorder and self.recorder is not None:
            self._debug("[start_recording()] Skipping 'recording_started' message - recorder is pre-initialized and ready")
        else:
            # Only send "recording_started" when initializing a new recorder
            self._send_message({"type": "status", "status": "recording_started", "timestamp": self._get_timestamp()})
            self._debug("[start_recording()] Sent recording_started status")
    
    def stop_recording(self):
        """Stop recording and finalize transcript file"""
        if not self.is_recording and self.recorder is None:
            return
        
        self.is_recording = False
        
        # Debug: Log preserve state
        self._debug(f"[stop_recording()] recorder exists: {self.recorder is not None}, _preserve_recorder: {self._preserve_recorder}")
        
        # Close recorder if it exists, but preserve it in pre-init mode
        # Defensive check: Only destroy if explicitly not in preserve mode
        if self.recorder:
            if self._preserve_recorder:
                # Preserve mode: Keep the recorder for reuse
                self._debug("[stop_recording()] Preserving recorder (pre-init mode)")
                # Do NOT set self.recorder = None
                # Validate that recorder is still set after preserving
                if self.recorder is None:
                    self._warn("[stop_recording()] ERROR: Recorder became None in preserve mode - this should not happen!")
                else:
                    self._debug(f"[stop_recording()] Recorder successfully preserved: {self.recorder is not None}")
            else:
                # Normal mode: Destroy the recorder
                self._debug("[stop_recording()] Destroying recorder (not in preserve mode)")
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
    
    def initialize_recorder(self):
        """Initialize the AudioToTextRecorder if not already initialized"""
        self._debug(f"[initialize_recorder()] Called - recorder exists: {self.recorder is not None}, _preserve_recorder: {self._preserve_recorder}")
        if self.recorder is not None:
            # Already initialized
            self._debug("[initialize_recorder()] Recorder already exists, returning early")
            self._send_message({"type": "status", "status": "recorder_ready", "timestamp": self._get_timestamp()})
            return
        
        # Track all real-time updates to prevent losing any text
        self.realtime_buffer = []
        self.last_audio_detected = time.monotonic()
        
        def on_realtime_update(text):
            """Callback for real-time transcription updates"""
            import threading
            thread_id = threading.current_thread().ident
            thread_name = threading.current_thread().name
            self._info(f"[THREAD CHECK] Realtime update - Thread ID: {thread_id}, Name: {thread_name}")
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

            # The realtime transcription contains only the current segment's partial text
            # We need to combine it with completed sentences from previous segments
            # Strategy: Join completed sentences with spaces, then append current partial
            completed_text = " ".join(self.full_transcript).strip() if self.full_transcript else ""
            
            # Combine completed sentences with current partial
            if completed_text:
                # Check if the realtime text is already included in completed text (shouldn't happen, but safety check)
                if cleaned.lower() not in completed_text.lower():
                    full_display = f"{completed_text} {cleaned}".strip()
                else:
                    # Realtime text is already in completed - just use completed (shouldn't happen normally)
                    full_display = completed_text
            else:
                # No completed sentences yet - just use realtime text
                full_display = cleaned

            emit_timestamp_iso = self._get_timestamp()
            emit_epoch_ms = self._get_epoch_ms()

            self._info(f"Cleaned text: {cleaned}")
            self._info(f"Full display: {full_display}")

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
            self._schedule_suggestion_timeout(cleaned, full_display)
        
        # Store callback for use in run()
        self._on_realtime_update = on_realtime_update
        
        # Define on_transcription_complete callback BEFORE creating recorder
        def on_transcription_complete(text):
            """Callback for completed transcriptions - called by _process_transcription_queue when silence is detected"""
            self._info(f"[CALLBACK] on_transcription_complete invoked with text length: {len(text) if text else 0}")
            if text:
                self._info(f"[CALLBACK] Text preview: {text[:50]}...")
            else:
                self._info("[CALLBACK] Text is empty or None")
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
                        # Merge with realtime buffer: Complete transcription is more accurate,
                        # but realtime buffer may have additional words spoken after silence detection
                        # Strategy: Append complete transcription, then append any additional words from realtime buffer
                        if self.realtime_buffer:
                            # Get the most recent realtime buffer entry
                            buffer_text = self.realtime_buffer[-1] if self.realtime_buffer else ""
                            
                            # If buffer has additional content not in complete transcription, append it
                            if buffer_text and buffer_text.strip():
                                # Check if buffer text contains the complete transcription plus more
                                if cleaned.lower() in buffer_text.lower():
                                    # Buffer has complete transcription plus additional words - use buffer
                                    cleaned = buffer_text
                                elif buffer_text.lower() not in cleaned.lower():
                                    # Buffer has different/additional content - append it
                                    cleaned = f"{cleaned} {buffer_text}".strip()
                        
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
        
        # Create recorder with callbacks
        try:
            recorder_config = {
                'spinner': False,
                'model': self.model,
                'realtime_model_type': self.realtime_model,
                'language': self.language,
                'silero_sensitivity': 0.15,
                'webrtc_sensitivity': 0,
                'post_speech_silence_duration': 2.5,
                'min_length_of_recording': 0.3,
                'min_gap_between_recordings': 0.05,
                'enable_realtime_transcription': True,
                'realtime_processing_pause': 0.01,
                'on_realtime_transcription_update': on_realtime_update,
                'on_transcription_complete': on_transcription_complete,
                'silero_deactivity_detection': True,
                'early_transcription_on_silence': 1.8,
                'silero_use_onnx': True,
                'beam_size': 10,
                'beam_size_realtime': 7,
                'no_log_file': False,
                'faster_whisper_vad_filter': True,
                'use_microphone': True,
                'download_root': None,
            }
            
            self._send_message({"type": "status", "status": "initializing_recorder", "timestamp": self._get_timestamp()})
            start = time.time()
            self._debug("Creating AudioToTextRecorder with config")
            self._debug(f"Config: model={self.model}, realtime_model={self.realtime_model}, language={self.language}")
            
            try:
                self._debug(f"Creating recorder with VAD: silero={recorder_config['silero_sensitivity']}, webrtc={recorder_config['webrtc_sensitivity']}")
                
                self.recorder = AudioToTextRecorder(**recorder_config)
                self.recorder_config = recorder_config
                self._debug("AudioToTextRecorder created successfully and ready to listen")
                
                try:
                    import pyaudio
                    end = time.time()
                    self._info(f"Time Taken to initialize AudioToTextRecorder: {end-start} seconds")
                    start = time.time()
                    p_verify = pyaudio.PyAudio()
                    end = time.time()
                    self._info(f"Time Taken to initialize pyaudio: {end-start} seconds")
                    start = time.time()
                    default_input = p_verify.get_default_input_device_info()
                    end = time.time()
                    self._info(f"Time Taken to get default input device info: {end-start} seconds")
                    self._info(f"[PLATFORM] Recorder active on microphone: {default_input['name']} (index={default_input['index']})")
                    p_verify.terminate()
                except Exception as verify_error:
                    self._warn(f"[PLATFORM] Could not verify microphone after recorder creation: {verify_error}")
                
            except Exception as init_error:
                self._send_error(f"Error creating recorder: {init_error}")
                self._warn("Recorder creation failed on Apple Silicon")
                self._info("[PLATFORM] Common fixes for M1/M2/M3:")
                self._info("  1. Install PortAudio: brew install portaudio")
                self._info("  2. Reinstall PyAudio: pip install --force-reinstall pyaudio")
                self._info("  3. Check microphone permissions in System Settings")
                self._info("  4. Try running from Terminal (not IDE) to get permission prompt")
                import traceback
                self._debug(f"Init traceback: {traceback.format_exc()}")
                raise
            
            self._send_message({"type": "status", "status": "recorder_ready", "timestamp": self._get_timestamp()})
            
        except Exception as init_error:
            self._send_error(f"Error creating recorder: {init_error}")
            raise

    def run(self):
        """Main run loop - continuously transcribe audio"""
        # Debug: Log recorder state when run() is called
        self._debug(f"[run()] Called - recorder exists: {self.recorder is not None}, _preserve_recorder: {self._preserve_recorder}")
        
        # Initialize recorder if not already initialized
        # Only initialize if recorder is None AND we're not in preserve mode (or if in preserve mode but recorder should exist)
        if self.recorder is None:
            if self._preserve_recorder:
                # In preserve mode, recorder should not be None - this is an error condition
                self._warn("[run()] ERROR: Recorder is None but _preserve_recorder is True - this should not happen!")
                self._debug("[run()] Attempting to initialize recorder anyway...")
            self._debug("[run()] Recorder is None, calling initialize_recorder()")
            self.initialize_recorder()
        else:
            self._debug("[run()] Recorder already exists, skipping initialization")
            # If recorder already exists, send "recorder_ready" status immediately
            # (skip "initializing_recorder" since we didn't initialize)
            self._send_message({"type": "status", "status": "recorder_ready", "timestamp": self._get_timestamp()})
            self._debug("[run()] Sent recorder_ready status for existing recorder")
        
        # Use stored callback or define it if not pre-initialized
        if hasattr(self, '_on_realtime_update'):
            on_realtime_update = self._on_realtime_update
        else:
            # Define callback (shouldn't happen if initialize_recorder was called)
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

                display_segments = self.full_transcript + ([cleaned] if cleaned else [])
                full_display = "\n".join(display_segments).strip()

                emit_timestamp_iso = self._get_timestamp()
                emit_epoch_ms = self._get_epoch_ms()

                self._info(f"Cleaned text: {cleaned}")
                self._info(f"Full display: {full_display}")

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
                self._schedule_suggestion_timeout(cleaned, full_display)
        
        # Platform detection and M1-specific diagnostics
        # import platform
        # system_info = platform.system()
        # machine = platform.machine()
        # self._info(f"[PLATFORM] System: {system_info}, Architecture: {machine}")
        
        # Check for Apple Silicon (M1/M2/M3/M4)
        # is_apple_silicon = machine == 'arm64' and system_info == 'Darwin'
        # if is_apple_silicon:
        #     self._info("[PLATFORM] Detected Apple Silicon (M1/M2/M3/M4) - checking compatibility...")
            
        #     # Test microphone access before creating recorder
        #     try:
        #         import pyaudio
        #         p_test = pyaudio.PyAudio()
        #         try:
        #             default_input = p_test.get_default_input_device_info()
        #             self._info(f"[PLATFORM] Microphone detected: {default_input['name']} (index={default_input['index']}, sample_rate={default_input['defaultSampleRate']} Hz)")
                    
        #             # Try to open a test stream
        #             test_stream = p_test.open(
        #                 format=pyaudio.paInt16,
        #                 channels=1,
        #                 rate=16000,
        #                 input=True,
        #                 frames_per_buffer=1024
        #             )
        #             self._info("[PLATFORM] Microphone stream test passed on Apple Silicon")
        #             test_stream.stop_stream()
        #             test_stream.close()
        #         except Exception as mic_error:
        #             self._warn(f"[PLATFORM] Microphone access test failed on Apple Silicon: {mic_error}")
        #             self._warn("[PLATFORM] Possible causes: permissions not granted, PyAudio mismatch, or PortAudio issue")
        #         finally:
        #             p_test.terminate()
        #     except ImportError:
        #         self._debug("[PLATFORM] PyAudio not available for microphone test")
            # except Exception as e:
            #     self._warn(f"[PLATFORM] Error testing microphone: {e}")
        
        # Create recorder with callbacks
        # The recorder is now initialized in initialize_recorder()
        # self.recorder = AudioToTextRecorder(**recorder_config)
        # self.recorder_config = recorder_config
        # self._debug("AudioToTextRecorder created successfully and ready to listen")
        
        # if is_apple_silicon:
        #     try:
        #         import pyaudio
        #         end= time.time()
        #         self._info(f"Time Taken to initialize AudioToTextRecorder: {end-start} seconds")
        #         start = time.time()
        #         p_verify = pyaudio.PyAudio()
        #         end = time.time()
        #         self._info(f"Time Taken to initialize pyaudio: {end-start} seconds")
        #         start = time.time()
        #         default_input = p_verify.get_default_input_device_info()
        #         end = time.time()
        #         self._info(f"Time Taken to get default input device info: {end-start} seconds")
        #         self._info(f"[PLATFORM] Recorder active on microphone: {default_input['name']} (index={default_input['index']})")
        #         p_verify.terminate()
        #     except Exception as verify_error:
        #         self._warn(f"[PLATFORM] Could not verify microphone after recorder creation: {verify_error}")
        
        # Start recording if not already started
        # Note: In pre-init mode, start_recording() is called by stdin_listener before run() is called
        # So we only call it here if it wasn't already called
        if not self.is_recording:
            self._debug("[run()] Starting recording (is_recording was False)")
            self.start_recording()
        else:
            self._debug("[run()] Recording already active (is_recording was True)")
        
        # Send "ready" status to indicate transcription loop is active
        # This comes after "recorder_ready" which was sent earlier (line 550)
        self._send_message({"type": "status", "status": "ready", "timestamp": self._get_timestamp()})
        self._debug("[run()] Sent 'ready' status - transcription loop active")
        self._debug("[run()] Callbacks will fire when speech is detected")
        
        # Listen continuously
        import threading
        import queue
        
        # Use shared stop queue if available (pre-init mode), otherwise create new one
        if self._shared_stop_queue is not None:
            stop_queue = self._shared_stop_queue
            self._debug("Using shared stop queue from pre-init mode")
        else:
            stop_queue = queue.Queue()
            
            # Thread to listen for stop commands (only if not in pre-init mode)
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
        try: 
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
                    # Only log VAD settings if recorder_config is available
                    if hasattr(self, 'recorder_config') and self.recorder_config:
                        self._debug(f"VAD Settings: silero={self.recorder_config.get('silero_sensitivity')}, webrtc={self.recorder_config.get('webrtc_sensitivity')}")
                    else:
                        self._debug("VAD Settings: using pre-initialized recorder configuration")
                    
                    # Continuous recording mode: Don't call blocking text() method
                    # Instead, recorder runs continuously and:
                    #   - on_realtime_transcription_update is called continuously (partial updates)
                    #   - on_transcription_complete is called by _process_transcription_queue when silence is detected
                    # This ensures no speech is missed and recording never stops
                    if iteration % 50 == 0:  # Only log every 50 iterations (~5 seconds) to reduce spam
                        self._debug(f"Continuous recording mode - iteration {iteration}")
                        self._info(f"[CONTINUOUS] Recording continuously - realtime updates active, complete transcriptions triggered on silence")
                    
                    # Just wait a bit to allow callbacks to process, then continue loop
                    # The recorder is already running and processing audio
                    time.sleep(0.1)  # Small sleep to prevent tight loop
                    
                    # Reset iteration tracking periodically (every 10 iterations = ~1 second)
                    if iteration % 10 == 0:
                        self.current_iteration_start = None
                        self.current_iteration_wallclock_iso = None
                        self.current_iteration_start_epoch_ms = None
                        self.current_iteration_monotonic_ms = None
                        self.first_partial_latency_ms = None
                        self.partial_update_count = 0
                        self.partial_logged = False
                    
                    # Reset error counter on success
                    consecutive_errors = 0
                    self._debug(f"Iteration {iteration} completed successfully - continuous recording mode")
                    
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
            # Verify recorder state after stop_recording() if in preserve mode
            if self._preserve_recorder:
                if self.recorder is None:
                    self._warn("[run()] ERROR: Recorder is None after stop_recording() but _preserve_recorder is True!")
                else:
                    self._debug(f"[run()] Recorder preserved after stop: {self.recorder is not None}")
    
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
            parser.add_argument('--pre-init', action='store_true', help='Pre-initialize recorder and wait for start command')
            
            args = parser.parse_args()
            
            service = RealtimeSTTService(
                transcript_file_path=args.transcript_file,
                model=args.model,
                realtime_model=args.realtime_model,
                language=args.language
            )
            
            if args.pre_init:
                # Pre-initialize recorder but don't start recording yet
                service._preserve_recorder = True  # Preserve recorder for multiple start/stop cycles
                service._debug(f"[main()] Pre-init mode enabled - Set _preserve_recorder = True")
                service._debug(f"[main()] Initializing recorder before run() is called")
                service.initialize_recorder()
                service._debug(f"[main()] Recorder initialized - recorder exists: {service.recorder is not None}")
                # Wait for start command on stdin
                import threading
                import queue
                stop_queue = queue.Queue()
                service._shared_stop_queue = stop_queue  # Share stop queue with run()
                run_thread = None
                
                def stdin_listener():
                    nonlocal run_thread
                    try:
                        while True:
                            line = sys.stdin.readline()
                            if not line:
                                break
                            try:
                                command = json.loads(line.strip())
                                action = command.get("action")
                                service._debug(f"[stdin_listener] Received command: {action}")
                                
                                if action == "start":
                                    # Check if already running
                                    if run_thread and run_thread.is_alive():
                                        service._debug("[stdin_listener] Recording already in progress, ignoring start command")
                                        continue
                                    
                                    service._debug(f"[stdin_listener] Starting recording - recorder exists: {service.recorder is not None}, _preserve_recorder: {service._preserve_recorder}")
                                    
                                    # If recorder is pre-initialized and ready, send "recorder_ready" IMMEDIATELY
                                    # This ensures the UI status updates as quickly as possible, before any other processing
                                    if service._preserve_recorder and service.recorder is not None:
                                        service._debug("[stdin_listener] Recorder is pre-initialized, sending recorder_ready immediately")
                                        service._send_message({"type": "status", "status": "recorder_ready", "timestamp": service._get_timestamp()})
                                    
                                    service.start_recording()
                                    # Don't send "ready" status here - let run() handle all status messages in correct sequence
                                    
                                    # Run in a separate thread so we can handle multiple cycles
                                    def run_transcription():
                                        try:
                                            service._debug("[run_transcription] Thread started, calling service.run()")
                                            service.run()
                                        except Exception as e:
                                            service._send_error(f"Error in transcription loop: {e}")
                                        finally:
                                            # Reset run_thread when transcription loop exits
                                            nonlocal run_thread
                                            run_thread = None
                                            service._debug("[run_transcription] Transcription loop exited, ready for next start")
                                    
                                    run_thread = threading.Thread(target=run_transcription, daemon=True)
                                    run_thread.start()
                                    service._debug(f"[stdin_listener] Started run_transcription thread")
                                    
                                elif action == "stop":
                                    service._debug("[stdin_listener] Received stop command, putting in queue")
                                    # Put stop command in queue for run() to handle
                                    stop_queue.put("stop")
                            except (json.JSONDecodeError, ValueError):
                                pass
                    except:
                        pass
                
                stdin_thread = threading.Thread(target=stdin_listener, daemon=True)
                stdin_thread.start()
                
                # Keep the process alive
                try:
                    while True:
                        time.sleep(0.1)
                        if not stdin_thread.is_alive():
                            break
                except KeyboardInterrupt:
                    pass
            else:
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
