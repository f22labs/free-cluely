#!/bin/bash
# Debug script to test transcription

cd "$(dirname "$0")"

echo "=== Testing RealtimeSTT Service ===" >&2
echo "Starting Python service with debug output..." >&2

python3 realtime_stt_service.py \
  --transcript-file /tmp/debug_transcript.txt \
  --model tiny.en \
  --realtime-model tiny.en \
  --language en \
  2>&1 | tee /tmp/debug_output.log

echo ""
echo "=== Transcription file content ===" >&2
cat /tmp/debug_transcript.txt

