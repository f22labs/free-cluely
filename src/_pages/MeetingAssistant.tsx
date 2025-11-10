import React, { useState, useEffect, useRef, useCallback } from "react"
import {
  Toast,
  ToastTitle,
  ToastDescription,
  ToastVariant,
  ToastMessage
} from "../components/ui/toast"
import ModelSelector from "../components/ui/ModelSelector"
import type {
  MeetingSuggestionMetrics,
  RealtimeCompleteMetrics,
  RealtimePartialMetrics
} from "../types/electron"

interface MeetingAssistantProps {
  setView: React.Dispatch<React.SetStateAction<"queue" | "solutions" | "debug" | "meeting">>
}

interface Suggestion {
  text: string
  timestamp: number
  type: "response" | "question" | "negotiation"
}

const MeetingAssistant: React.FC<MeetingAssistantProps> = ({ setView }) => {
  const [toastOpen, setToastOpen] = useState(false)
  const [toastMessage, setToastMessage] = useState<ToastMessage>({
    title: "",
    description: "",
    variant: "neutral"
  })

  const [systemPrompt, setSystemPrompt] = useState("")
  const [isRecording, setIsRecording] = useState(false)
  const [transcript, setTranscript] = useState("")
  const [suggestions, setSuggestions] = useState<Suggestion[]>([])
  const [isGenerating, setIsGenerating] = useState(false)
  const [currentModel, setCurrentModel] = useState<{ provider: string; model: string }>({ 
    provider: "gemini", 
    model: "gemini-2.0-flash" 
  })
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  
  const transcriptRef = useRef<string>("")
  const suggestionTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const contentRef = useRef<HTMLDivElement>(null)
  const suggestionsEndRef = useRef<HTMLDivElement>(null)
  const lastProcessedTranscriptRef = useRef<string>("")
  const lastSuggestionTimeRef = useRef<number>(0)
  const lastSuggestionTextRef = useRef<string>("")
  const rateLimitBackoffRef = useRef<number>(2000) // Adaptive delay, starts at 2 seconds
  const lastTranscriptionMetaRef = useRef<{
    iteration: number | null
    pythonLatencyMs: number | null
    electronReceivedAt: number
    eventReceivedAt: number
    eventReceivedPerf: number
    pythonCompletionEpochMs: number | null
    metrics?: RealtimeCompleteMetrics
  } | null>(null)
  const llmRequestMetaRef = useRef<{ startPerf: number; startEpochMs: number } | null>(null)
  const pendingSuggestionDisplayRef = useRef<{ startPerf: number; startEpochMs: number } | null>(null)
  const lastSuggestionMetricsRef = useRef<MeetingSuggestionMetrics | undefined>(undefined)
  const previousSuggestionCountRef = useRef<number>(0)

  // Component mount logging
  useEffect(() => {
    console.log("[MeetingAssistant] Component mounted/rendered")
  }, [])

  // Load current model configuration on mount
  useEffect(() => {
    const loadCurrentModel = async () => {
      try {
        const config = await window.electronAPI.getCurrentLlmConfig()
        setCurrentModel({ provider: config.provider, model: config.model })
        console.log("[MeetingAssistant] Loaded model config:", config)
      } catch (error) {
        console.error('Error loading current model config:', error)
      }
    }
    loadCurrentModel()
  }, [])

  useEffect(() => {
    if (pendingSuggestionDisplayRef.current && suggestions.length > previousSuggestionCountRef.current) {
      const renderDelta = performance.now() - pendingSuggestionDisplayRef.current.startPerf
      console.log(`[MeetingAssistant][Metrics] Suggestion rendered in ${renderDelta.toFixed(2)}ms (total suggestions=${suggestions.length})`)
      pendingSuggestionDisplayRef.current = null
    }
    previousSuggestionCountRef.current = suggestions.length
  }, [suggestions])

  // Helper function to calculate text similarity (0-1, where 1 is identical)
  const calculateSimilarity = (text1: string, text2: string): number => {
    const normalize = (text: string) => text.toLowerCase().trim().replace(/[^\w\s]/g, '')
    const words1 = normalize(text1).split(/\s+/).filter(w => w.length > 0)
    const words2 = normalize(text2).split(/\s+/).filter(w => w.length > 0)
    
    if (words1.length === 0 || words2.length === 0) return 0
    
    // Calculate Jaccard similarity (intersection over union)
    const set1 = new Set(words1)
    const set2 = new Set(words2)
    const intersection = new Set([...set1].filter(x => set2.has(x)))
    const union = new Set([...set1, ...set2])
    
    return intersection.size / union.size
  }

  // Helper function to check if transcript has meaningful new content
  const hasNewContent = (newTranscript: string, lastProcessed: string): boolean => {
    if (!lastProcessed) {
      console.log("[MeetingAssistant] hasNewContent: No last processed, allowing (first time)")
      return true
    }
    
    // Normalize both transcripts
    const normalize = (text: string) => text.toLowerCase().trim()
    const newNorm = normalize(newTranscript)
    const lastNorm = normalize(lastProcessed)
    
    // Check if new transcript is significantly longer (new sentence added)
    const lengthDiff = newNorm.length - lastNorm.length
    console.log("[MeetingAssistant] hasNewContent check:", {
      lengthDiff,
      newLength: newNorm.length,
      lastLength: lastNorm.length,
      percentageGrowth: ((lengthDiff / lastNorm.length) * 100).toFixed(1) + "%"
    })
    
    // If transcript is shorter or same, definitely no new content
    if (lengthDiff <= 0) {
      console.log("[MeetingAssistant] hasNewContent: No length increase, rejecting")
      return false
    }
    
    // If it's a small update (< 15 chars), likely just a partial word/sentence correction
    if (lengthDiff < 15) {
      console.log("[MeetingAssistant] hasNewContent: Too small increase (< 15 chars), rejecting")
      return false
    }
    
    // Check if new transcript contains the old one (it's an extension)
    if (newNorm.includes(lastNorm)) {
      // Extract the new part
      const newPart = newNorm.substring(lastNorm.length).trim()
      console.log("[MeetingAssistant] New part extracted:", newPart.substring(0, 80))
      
      // More lenient: require at least 15 chars of new content (about 2-3 words)
      // This allows suggestions to generate more frequently during conversations
      if (newPart.length >= 15) {
        console.log("[MeetingAssistant] hasNewContent: New part is sufficient (", newPart.length, "chars), allowing")
        return true
      } else {
        console.log("[MeetingAssistant] hasNewContent: New part too short (", newPart.length, "chars), rejecting")
        return false
      }
    }
    
    // If transcript structure changed significantly (not just an extension), allow it
    console.log("[MeetingAssistant] hasNewContent: Transcript structure changed, allowing")
    return true
  }

  const generateSuggestion = useCallback(async (currentTranscript: string) => {
    console.log("[MeetingAssistant] generateSuggestion called:", {
      hasSystemPrompt: !!systemPrompt.trim(),
      hasTranscript: !!currentTranscript.trim(),
      isGenerating
    })
    
    if (!systemPrompt.trim() || !currentTranscript.trim() || isGenerating) {
      console.log("[MeetingAssistant] Skipping suggestion generation - missing requirements")
      return
    }

    const requestEpochMs = Date.now()
    const requestPerfMs = performance.now()
    const transcriptionMeta = lastTranscriptionMetaRef.current
    if (transcriptionMeta) {
      const timeSinceEventMs = requestPerfMs - transcriptionMeta.eventReceivedPerf
      const timeSinceElectronMs = requestEpochMs - transcriptionMeta.electronReceivedAt
      const timeSincePythonMs = transcriptionMeta.pythonCompletionEpochMs != null
        ? requestEpochMs - transcriptionMeta.pythonCompletionEpochMs
        : null
      console.log("[MeetingAssistant][Metrics] Preparing suggestion request:", {
        iteration: transcriptionMeta.iteration,
        pythonLatencyMs: transcriptionMeta.pythonLatencyMs,
        timeSinceRendererEventMs: Number.isFinite(timeSinceEventMs) ? timeSinceEventMs.toFixed(2) : null,
        timeSinceElectronMs,
        timeSincePythonCompleteMs: timeSincePythonMs
      })
    } else {
      console.log("[MeetingAssistant][Metrics] No transcription metadata available for suggestion request.")
    }
    llmRequestMetaRef.current = { startPerf: requestPerfMs, startEpochMs: requestEpochMs }
    lastSuggestionMetricsRef.current = undefined

    console.log("[MeetingAssistant] Starting suggestion generation...")
    setIsGenerating(true)
    try {
      const suggestion = await window.electronAPI.generateMeetingSuggestion(
        currentTranscript,
        systemPrompt
      )

      const responseEpochMs = Date.now()
      const responsePerfMs = performance.now()
      const requestMeta = llmRequestMetaRef.current
      const rendererRoundTripPerfMs = requestMeta ? responsePerfMs - requestMeta.startPerf : 0
      const rendererRoundTripEpochMs = requestMeta ? responseEpochMs - requestMeta.startEpochMs : 0
      console.log("[MeetingAssistant][Metrics] Suggestion response timing:", {
        rendererRoundTripPerfMs: rendererRoundTripPerfMs.toFixed(2),
        rendererRoundTripEpochMs,
        iteration: transcriptionMeta?.iteration ?? lastTranscriptionMetaRef.current?.iteration ?? null
      })
      llmRequestMetaRef.current = null
      
      console.log("[MeetingAssistant] Received suggestion:", suggestion)
      if (suggestion?.metrics) {
        lastSuggestionMetricsRef.current = suggestion.metrics
        console.log("[MeetingAssistant][Metrics] LLM metrics payload:", suggestion.metrics)
      } else {
        lastSuggestionMetricsRef.current = undefined
      }
      
      // Reset timer and gradually reduce backoff delay on success
      lastSuggestionTimeRef.current = Date.now()
      // Gradually reduce backoff delay (but not below 2 seconds)
      if (rateLimitBackoffRef.current > 2000) {
        rateLimitBackoffRef.current = Math.max(rateLimitBackoffRef.current * 0.9, 2000)
        console.log(`[MeetingAssistant] Success - reducing delay to ${rateLimitBackoffRef.current}ms`)
      }
      
      if (suggestion && suggestion.text) {
        const suggestionText = suggestion.text.trim()
        
        // Check similarity with last suggestion - this is the primary filter
        if (lastSuggestionTextRef.current) {
          const similarity = calculateSimilarity(suggestionText, lastSuggestionTextRef.current)
          
          // If similarity is too high (>= 0.65), it's likely a duplicate - skip it
          // Using 0.65 instead of 0.7 for stricter filtering
          if (similarity >= 0.65) {
            console.log("[MeetingAssistant] Skipping duplicate suggestion (similarity:", similarity.toFixed(2), ")")
            console.log("[MeetingAssistant] Last:", lastSuggestionTextRef.current.substring(0, 50) + "...")
            console.log("[MeetingAssistant] New:", suggestionText.substring(0, 50) + "...")
            setIsGenerating(false)
            return
          }
        }
        
        // Also check against all existing suggestions before creating the new one
        let appended = false
        setSuggestions(prev => {
          // Check if this suggestion is too similar to any existing suggestion
          const isDuplicate = prev.some(existing => {
            const similarity = calculateSimilarity(suggestionText, existing.text)
            return similarity >= 0.65 // Stricter threshold
          })
          
          if (isDuplicate) {
            console.log("[MeetingAssistant] Skipping duplicate suggestion (similar to existing)")
            setIsGenerating(false)
            return prev // Return unchanged state
          }
          
          // Valid new suggestion - create it
          const newSuggestion: Suggestion = {
            text: suggestionText,
            timestamp: Date.now(),
            type: suggestion.type || "response"
          }
          appended = true
          
          // Update refs
          lastSuggestionTextRef.current = suggestionText
          lastSuggestionTimeRef.current = Date.now()
          lastProcessedTranscriptRef.current = currentTranscript
          
          // Scroll to bottom when new suggestion is added
          setTimeout(() => {
            suggestionsEndRef.current?.scrollIntoView({ behavior: "smooth" })
          }, 100)
          
          // Only add if it's different from the last one
          return [...prev, newSuggestion]
        })
        if (appended) {
          pendingSuggestionDisplayRef.current = {
            startPerf: performance.now(),
            startEpochMs: Date.now()
          }
          console.log("[MeetingAssistant][Metrics] Queued suggestion display timing measurement")
        }
      }
    } catch (error: any) {
      llmRequestMetaRef.current = null
      console.error("Error generating suggestion:", error)
      
      // Handle rate limit errors with user-friendly message
      const errorMessage = error.message || String(error)
      if (errorMessage.includes("429") || 
          errorMessage.includes("Too Many Requests") || 
          errorMessage.includes("rate limit") ||
          errorMessage.includes("Resource exhausted")) {
        showToast(
          "Rate Limit Exceeded", 
          "API rate limit reached. The system will automatically retry with backoff. Please wait...", 
          "error"
        )
        // Increase delay for next attempts (adaptive backoff)
        rateLimitBackoffRef.current = Math.min(rateLimitBackoffRef.current * 1.5, 10000) // Max 10 seconds
        console.log(`[MeetingAssistant] Rate limit hit - increasing delay to ${rateLimitBackoffRef.current}ms`)
        // Reset timer to enforce longer wait
        lastSuggestionTimeRef.current = Date.now()
      } else if (errorMessage.includes("quota") || errorMessage.includes("Quota exceeded")) {
        showToast(
          "API Quota Exceeded", 
          "Your API quota has been exceeded. Please check your API usage limits.", 
          "error"
        )
      } else {
        showToast("Error", "Failed to generate suggestion: " + errorMessage, "error")
      }
    } finally {
      setIsGenerating(false)
      if (lastSuggestionMetricsRef.current) {
        console.log("[MeetingAssistant][Metrics] Latest suggestion metrics summary:", lastSuggestionMetricsRef.current)
      }
    }
  }, [systemPrompt, isGenerating])

  // Listen to real-time transcription updates (only for display, not for suggestions)
  useEffect(() => {
    console.log("[MeetingAssistant] Setting up transcription listeners")
    
    const unsubscribeUpdate = window.electronAPI.onRealtimeTranscriptionUpdate?.((data: { text: string; fullTranscript: string | null; metrics?: RealtimePartialMetrics }) => {
      console.log("[MeetingAssistant] Received realtime update:", data)
      if (data && (data.fullTranscript || data.text)) {
        const fullText = data.fullTranscript || data.text || ""
        transcriptRef.current = fullText
        setTranscript(fullText)
        console.log("[MeetingAssistant] Updated transcript:", fullText.substring(0, 50) + "...")
        // DO NOT generate suggestions on real-time updates - only on complete sentences
        if (data.metrics) {
          const elapsedSinceEmit = data.metrics.electron_emit_epoch_ms
            ? Date.now() - data.metrics.electron_emit_epoch_ms
            : null
          console.log("[MeetingAssistant][Metrics][Partial]", {
            iteration: data.metrics.iteration ?? null,
            partialIndex: data.metrics.partial_index ?? null,
            latencyMs: data.metrics.latency_ms ?? null,
            firstPartialLatencyMs: data.metrics.first_partial_latency_ms ?? null,
            electronEmitTimestamp: data.metrics.electron_emit_timestamp ?? null,
            rendererLatencyMs: elapsedSinceEmit
          })
        }
      }
    })

    const unsubscribeComplete = window.electronAPI.onRealtimeTranscriptionComplete?.((data: { text: string; fullTranscript: string; metrics?: RealtimeCompleteMetrics }) => {
      console.log("[MeetingAssistant] Received transcription complete:", data)
      if (data && (data.fullTranscript || data.text)) {
        const fullText = data.fullTranscript || data.text || ""
        transcriptRef.current = fullText
        setTranscript(fullText)
        console.log("[MeetingAssistant] Updated transcript (complete):", fullText.substring(0, 50) + "...")
        const nowEpoch = Date.now()
        const nowPerf = performance.now()

        if (data.metrics) {
          const rendererLag = data.metrics.electron_received_epoch_ms != null
            ? nowEpoch - data.metrics.electron_received_epoch_ms
            : 0
          const pythonToRenderer = data.metrics.python_to_electron_ms ?? null
          console.log("[MeetingAssistant][Metrics][Complete]", {
            iteration: data.metrics.iteration ?? null,
            pythonLatencyMs: data.metrics.transcription_latency_ms ?? null,
            pythonToRendererMs: pythonToRenderer,
            rendererLagMs: rendererLag,
            fallbackUsed: data.metrics.fallback_used ?? false,
            partialUpdates: data.metrics.partial_update_count ?? null
          })
          lastTranscriptionMetaRef.current = {
            iteration: data.metrics.iteration ?? null,
            pythonLatencyMs: data.metrics.transcription_latency_ms ?? null,
            electronReceivedAt: data.metrics.electron_received_epoch_ms ?? nowEpoch,
            eventReceivedAt: nowEpoch,
            eventReceivedPerf: nowPerf,
            pythonCompletionEpochMs: data.metrics.python_completion_epoch_ms ?? null,
            metrics: data.metrics
          }
        } else {
          console.log("[MeetingAssistant][Metrics][Complete] No metrics payload received for this event.")
          lastTranscriptionMetaRef.current = {
            iteration: null,
            pythonLatencyMs: null,
            electronReceivedAt: nowEpoch,
            eventReceivedAt: nowEpoch,
            eventReceivedPerf: nowPerf,
            pythonCompletionEpochMs: null,
            metrics: undefined
          }
        }
        
        // Only generate suggestion if:
        // 1. There's meaningful new content
        // 2. Enough time has passed since last suggestion (at least 5 seconds)
        // 3. We're not already generating
        const now = Date.now()
        const timeSinceLastSuggestion = now - lastSuggestionTimeRef.current
        // Use adaptive delay - increases after rate limit errors
        const MIN_TIME_BETWEEN_SUGGESTIONS = rateLimitBackoffRef.current
        
        const hasNew = hasNewContent(fullText, lastProcessedTranscriptRef.current)
        
        console.log("[MeetingAssistant] Checking suggestion generation:", {
          hasNewContent: hasNew,
          timeSinceLast: timeSinceLastSuggestion,
          isGenerating,
          transcriptLength: fullText.length,
          lastProcessedLength: lastProcessedTranscriptRef.current.length,
          willGenerate: hasNew && timeSinceLastSuggestion >= MIN_TIME_BETWEEN_SUGGESTIONS && !isGenerating
        })
        
        // IMPORTANT: Update lastProcessedTranscriptRef when we detect new content
        // This prevents getting stuck checking against the same old transcript
        // We update it here so the next check will be against this new transcript
        if (hasNew) {
          // Only update if we're not going to generate (to avoid double update)
          // If we are generating, let generateSuggestion update it after adding the suggestion
          if (!(timeSinceLastSuggestion >= MIN_TIME_BETWEEN_SUGGESTIONS && !isGenerating)) {
            lastProcessedTranscriptRef.current = fullText
            console.log("[MeetingAssistant] Updated lastProcessedTranscriptRef (no generation):", fullText.substring(0, 50) + "...")
          }
        }
        
        if (
          hasNew &&
          timeSinceLastSuggestion >= MIN_TIME_BETWEEN_SUGGESTIONS &&
          !isGenerating
        ) {
          console.log("[MeetingAssistant] ✅ All conditions met - Generating suggestion...")
          generateSuggestion(fullText)
        } else {
          const reasons = []
          if (!hasNew) reasons.push("no new content")
          if (timeSinceLastSuggestion < MIN_TIME_BETWEEN_SUGGESTIONS) reasons.push(`time threshold (${timeSinceLastSuggestion}ms < ${MIN_TIME_BETWEEN_SUGGESTIONS}ms)`)
          if (isGenerating) reasons.push("already generating")
          console.log("[MeetingAssistant] ❌ Not generating suggestion - reasons:", reasons.join(", "))
        }
      }
    })

    return () => {
      if (unsubscribeUpdate) unsubscribeUpdate()
      if (unsubscribeComplete) unsubscribeComplete()
      if (suggestionTimeoutRef.current) {
        clearTimeout(suggestionTimeoutRef.current)
      }
    }
  }, [systemPrompt, currentModel, isGenerating, generateSuggestion])

  const handleStartRecording = async () => {
    if (!systemPrompt.trim()) {
      showToast("System Prompt Required", "Please enter a system prompt before starting the meeting", "error")
      return
    }

    try {
      const result = await window.electronAPI.startRealTimeTranscription()
      if (result.success) {
        setIsRecording(true)
        setTranscript("")
        setSuggestions([])
        transcriptRef.current = ""
        // Reset tracking refs
        lastProcessedTranscriptRef.current = ""
        lastSuggestionTimeRef.current = 0
        lastSuggestionTextRef.current = ""
        rateLimitBackoffRef.current = 2000 // Reset to default 2 seconds
        showToast("Recording Started", "Meeting assistant is now active", "success")
      } else {
        showToast("Error", result.error || "Failed to start transcription", "error")
      }
    } catch (error: any) {
      showToast("Error", "Failed to start recording: " + error.message, "error")
    }
  }

  const handleStopRecording = async () => {
    try {
      const result = await window.electronAPI.stopRealTimeTranscription()
      if (result.success) {
        setIsRecording(false)
        showToast("Recording Stopped", "Transcription saved", "success")
      }
    } catch (error: any) {
      showToast("Error", "Failed to stop recording: " + error.message, "error")
    }
  }

  const showToast = (
    title: string,
    description: string,
    variant: ToastVariant
  ) => {
    setToastMessage({ title, description, variant })
    setToastOpen(true)
  }

  const handleModelChange = (provider: "ollama" | "gemini", model: string) => {
    setCurrentModel({ provider, model })
  }

  // Update dimensions
  useEffect(() => {
    const updateDimensions = () => {
      if (contentRef.current) {
        const height = contentRef.current.scrollHeight
        const width = contentRef.current.scrollWidth
        window.electronAPI.updateContentDimensions({ width, height })
      }
    }

    const resizeObserver = new ResizeObserver(updateDimensions)
    if (contentRef.current) {
      resizeObserver.observe(contentRef.current)
    }
    updateDimensions()

    return () => {
      resizeObserver.disconnect()
    }
  }, [suggestions, transcript, isSettingsOpen])

  return (
    <div 
      ref={contentRef} 
      className="min-h-0 w-full rounded-lg shadow-2xl border border-gray-700/50 overflow-hidden"
      style={{
        background: 'linear-gradient(135deg, rgba(17, 24, 39, 0.98) 0%, rgba(31, 41, 55, 0.98) 100%)',
        backdropFilter: 'blur(20px)',
        WebkitBackdropFilter: 'blur(20px)',
        position: 'relative',
        zIndex: 10
      }}
    >
      <div className="w-full h-full">
        <div className="px-4 py-3" style={{ background: 'rgba(17, 24, 39, 0.95)' }}>
          <Toast
            open={toastOpen}
            onOpenChange={setToastOpen}
            variant={toastMessage.variant}
            duration={3000}
          >
            <ToastTitle>{toastMessage.title}</ToastTitle>
            <ToastDescription>{toastMessage.description}</ToastDescription>
          </Toast>

          {/* Header with Close Button */}
          <div className="flex items-center justify-between mb-4 pb-3 border-b border-gray-600/70 bg-gray-800/30 px-2 py-2 rounded-md">
            <h2 className="text-xl font-bold text-white drop-shadow-lg">🤝 Meeting Assistant</h2>
            <div className="flex gap-2 items-center">
              <button
                className="bg-gray-700/50 hover:bg-gray-700/70 transition-colors rounded-md px-3 py-1.5 text-xs text-white/90 font-medium"
                onClick={() => setIsSettingsOpen(!isSettingsOpen)}
              >
                ⚙️ Models
              </button>
              <button
                className="bg-gray-700/50 hover:bg-gray-700/70 transition-colors rounded-md px-3 py-1.5 text-xs text-white/90 font-medium"
                onClick={() => setView("queue")}
              >
                ← Back
              </button>
              <button
                className="bg-red-500/80 hover:bg-red-500/90 transition-colors rounded-md px-3 py-1.5 text-xs text-white font-medium"
                onClick={() => setView("queue")}
                title="Close"
              >
                ✕
              </button>
            </div>
          </div>

          {/* Settings */}
          {isSettingsOpen && (
            <div className="mb-4">
              <ModelSelector onModelChange={handleModelChange} onChatOpen={() => {}} />
            </div>
          )}

          {/* System Prompt Input */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-white mb-2 drop-shadow">
              System Prompt (Context about you and the meeting)
            </label>
            <textarea
              className="w-full px-3 py-2 rounded-lg bg-gray-800/90 text-gray-100 placeholder-gray-500 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 border border-gray-600/70 shadow-lg resize-none backdrop-blur-sm"
              placeholder="Example: I am a CEO of a tech startup meeting with potential clients. The meeting is about discussing our new AI product, pricing, and implementation timeline. I need to be empathetic to client concerns while being technically accurate about our capabilities."
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              rows={4}
              disabled={isRecording}
            />
            <p className="text-xs text-gray-400 mt-1">
              Describe who you are, what the meeting is about, and how you want the AI to assist you.
            </p>
          </div>

          {/* Recording Controls */}
          <div className="mb-4 flex gap-2">
            <button
              className={`flex-1 px-4 py-3 rounded-lg text-sm font-semibold transition-colors shadow-lg ${
                isRecording
                  ? "bg-red-500 hover:bg-red-600 text-white"
                  : "bg-green-500 hover:bg-green-600 text-white"
              } disabled:opacity-50 disabled:cursor-not-allowed`}
              onClick={isRecording ? handleStopRecording : handleStartRecording}
              disabled={!systemPrompt.trim()}
            >
              {isRecording ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="animate-pulse">●</span> Stop Recording
                </span>
              ) : (
                "🎤 Start Meeting"
              )}
            </button>
          </div>

          {/* Transcript Display - Always visible */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-white mb-2 drop-shadow">
              📝 Live Transcript
            </label>
            <div className="w-full px-4 py-3 rounded-lg bg-gray-800/90 text-gray-100 text-sm border border-gray-600/70 shadow-lg max-h-48 overflow-y-auto backdrop-blur-sm min-h-[100px]">
              {transcript || (
                <span className="text-gray-400 italic">
                  {isRecording 
                    ? "Listening... Transcription will appear here as you speak." 
                    : "Start the meeting to see live transcription here."}
                </span>
              )}
            </div>
          </div>

          {/* Suggestions Display - Single scrollable box */}
          <div className="mb-4">
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-medium text-white drop-shadow">
                💡 AI Suggestion
              </label>
              {isGenerating && (
                <span className="text-xs text-gray-300 animate-pulse bg-blue-500/20 px-2 py-1 rounded">Generating...</span>
              )}
            </div>
            <div className="w-full rounded-lg bg-gray-800/90 border border-gray-600/70 shadow-lg max-h-64 overflow-y-auto backdrop-blur-sm">
              {suggestions.length === 0 ? (
                <div className="w-full px-4 py-6 text-gray-300 text-sm text-center bg-gray-700/20">
                  {isRecording
                    ? "AI suggestions will appear here as the conversation progresses..."
                    : "Start the meeting to receive AI suggestions"}
                </div>
              ) : (
                <div className="p-3 space-y-3">
                  {suggestions.map((suggestion, idx) => (
                    <div
                      key={idx}
                      className={`px-4 py-3 rounded-lg backdrop-blur-sm ${
                        idx === suggestions.length - 1
                          ? "bg-blue-600/40 border-2 border-blue-400/60 shadow-lg"
                          : "bg-gray-700/60 border border-gray-600/40"
                      }`}
                    >
                      <div className="flex items-start gap-3">
                        <span className="text-lg mt-0.5 flex-shrink-0 drop-shadow">
                          {suggestion.type === "negotiation" ? "💼" : 
                           suggestion.type === "question" ? "❓" : "💡"}
                        </span>
                        <div className="flex-1 min-w-0">
                          <p className="text-xs text-gray-300 mb-1.5 uppercase tracking-wide font-semibold">
                            {suggestion.type === "negotiation" ? "Negotiation" : 
                             suggestion.type === "question" ? "Question" : "Response"}
                          </p>
                          <p className="text-sm text-white leading-relaxed drop-shadow-sm">{suggestion.text}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                  <div ref={suggestionsEndRef} />
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default MeetingAssistant

