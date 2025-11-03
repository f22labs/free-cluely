# Troubleshooting Real-Time Transcription

## Issue: Speech Stops After First Line

If transcription works for the first sentence but then stops detecting speech, this is usually due to:

### 1. Model Too Large for Real-Time
**Problem**: `medium.en` (769MB) or larger models are too slow for real-time processing.

**Solution**: Use lighter models for real-time:
- ✅ **Recommended**: `base.en` or `small.en` for real-time
- ❌ **Too slow**: `medium.en`, `large-v2`, `large-v3` for real-time

### 2. Memory/Performance Issues
**Check**: Monitor CPU/RAM usage when recording

**Solution**: 
- Close other applications
- Use lighter models
- Ensure sufficient RAM (24GB should be fine)

### 3. Recorder Getting Stuck
**Check**: Look for debug messages in terminal:
```
[DEBUG] Loop iteration 1: Calling recorder.text()...
[DEBUG] recorder.text() returned...
[DEBUG] Loop iteration 2: Calling recorder.text()...  ← Should see this
```

**If iteration 2 never appears**: The recorder is stuck.

## Recommended Configuration

### For Best Balance:
```
Real-time model: base.en (74MB) - fast, good accuracy
Final model: large-v2 (1550MB) - best accuracy when sentence completes
```

### For Better Accuracy (if system can handle):
```
Real-time model: small.en (244MB) - still fast enough
Final model: large-v2 (1550MB) - best accuracy
```

### Avoid:
```
❌ medium.en for real-time - too slow
❌ large-v2/v3 for real-time - way too slow
```

## Debugging Steps

1. Check terminal output for `[DEBUG]` messages
2. Look for error messages after first transcription
3. Check if loop iterations continue (should see "Loop iteration 2", "Loop iteration 3", etc.)
4. If iterations stop, the recorder is stuck - try lighter model

