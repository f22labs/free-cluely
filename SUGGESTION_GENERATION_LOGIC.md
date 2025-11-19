# Meeting Assistant - Suggestion Generation Logic

## When Suggestions Are Generated

Suggestions are triggered when **ALL** of the following conditions are met:

### 1. **Event Type: Only on Complete Sentences**
- ✅ Triggers on: `transcription_complete` events (when a sentence finishes)
- ❌ Does NOT trigger on: `realtime_transcription_update` events (partial/real-time updates)

### 2. **New Content Detection** (`hasNewContent()`)
The transcript must have meaningful new content compared to the last processed transcript:

- **First time**: Always allows (no previous transcript to compare)
- **Length increase**: Must have at least **15 characters** of new content
- **New part extraction**: If transcript extends the previous one, the new part must be at least **15 characters** (about 2-3 words)

**Example:**
```
Previous: "Hello, how are you?"
New: "Hello, how are you? I'm calling about the product."
New part: "I'm calling about the product." (30 chars) ✅ ALLOWS

Previous: "Hello, how are you?"
New: "Hello, how are you? Hi"  
New part: "Hi" (2 chars) ❌ REJECTS (too short)
```

### 3. **Time Threshold**
- Must wait at least **2 seconds** since the last suggestion was generated
- This prevents rapid-fire suggestions during fast conversations

### 4. **Not Already Generating**
- Cannot generate a new suggestion while one is already being processed

### 5. **System Prompt Required**
- Must have a system prompt entered (cannot be empty)

### 6. **Duplicate Filtering** (After Generation)
After the AI generates a suggestion, it's checked for duplicates:
- **Similarity check**: If the new suggestion is ≥65% similar to the last suggestion, it's rejected
- **Existing suggestions check**: If it's ≥65% similar to any existing suggestion, it's rejected

## Current Thresholds (After Fixes)

| Threshold | Value | Purpose |
|-----------|-------|---------|
| **Minimum new content** | 15 characters | About 2-3 words |
| **Time between suggestions** | 2 seconds | Prevents spam |
| **Similarity threshold** | 65% | Filters duplicates |

## Why Suggestions Might Not Appear

1. **Too short sentences**: If each sentence adds < 15 characters, it won't trigger
2. **Too fast conversation**: If sentences complete within 2 seconds of each other, only the first will trigger
3. **Duplicate suggestions**: If AI generates similar suggestions, they're filtered out
4. **No system prompt**: Must enter context before starting
5. **Transcription not working**: Check if transcript is appearing in the UI

## Debugging

Check the browser console for these logs:
- `[MeetingAssistant] Received transcription complete` - Event received
- `[MeetingAssistant] hasNewContent check` - Content check details
- `[MeetingAssistant] Checking suggestion generation` - Condition evaluation
- `[MeetingAssistant] ✅ All conditions met` - Suggestion will generate
- `[MeetingAssistant] ❌ Not generating suggestion - reasons:` - Why it was rejected

## Example Flow

```
1. User speaks: "Hello, I'm interested in your product"
   → transcription_complete event fires
   → hasNewContent: true (first time, no previous)
   → Time check: OK (first suggestion)
   → ✅ GENERATES SUGGESTION

2. User speaks: "Can you tell me more about pricing?"
   → transcription_complete event fires  
   → hasNewContent: true (new part: "Can you tell me more about pricing?" = 40 chars)
   → Time check: OK (2+ seconds passed)
   → ✅ GENERATES SUGGESTION

3. User speaks: "Okay"
   → transcription_complete event fires
   → hasNewContent: false (new part: "Okay" = 4 chars < 15 chars)
   → ❌ REJECTS (too short)

4. User speaks: "What about the implementation timeline?"
   → transcription_complete event fires
   → hasNewContent: true (new part: "What about the implementation timeline?" = 38 chars)
   → Time check: ❌ Only 1 second passed (need 2 seconds)
   → ❌ REJECTS (time threshold)
```

