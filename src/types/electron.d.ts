export interface RealtimePartialMetrics {
  iteration?: number | null
  partial_index?: number | null
  latency_ms?: number | null
  first_partial_latency_ms?: number | null
  python_iteration_started_at?: string | null
  python_iteration_started_epoch_ms?: number | null
  python_emit_timestamp?: string | null
  python_emit_epoch_ms?: number | null
  electron_emit_timestamp?: string
  electron_emit_epoch_ms?: number
}

export interface RealtimeCompleteMetrics extends RealtimePartialMetrics {
  transcription_latency_ms?: number | null
  partial_update_count?: number | null
  python_completion_timestamp?: string | null
  python_completion_epoch_ms?: number | null
  electron_received_epoch_ms?: number
  python_to_electron_ms?: number | null
  fallback_used?: boolean
}

export interface RealtimeTimeoutMetrics extends RealtimePartialMetrics {
  timeoutElapsedMs?: number | null
  timeoutSeconds?: number | null
  timeoutSequence?: number | null
}

export interface MeetingSuggestionTimelineMetrics {
  iteration: number | null
  python_latency_ms: number | null
  python_first_partial_latency_ms: number | null
  python_to_electron_ms: number | null
  electron_received_epoch_ms: number
  electron_emit_timestamp: string
  renderer_request_epoch_ms: number
  transcription_to_request_ms: number
  transcription_to_request_python_ms: number | null
  fallback_used: boolean
  partial_update_count: number | null
}

export interface MeetingSuggestionMetrics {
  provider: "ollama" | "gemini"
  model: string
  attempts: number
  llm_duration_ms: number
  llm_started_at: string
  llm_completed_at: string
  llm_round_trip_duration_ms?: number
  renderer_request_epoch_ms?: number
  renderer_response_epoch_ms?: number
  transcription_timeline?: MeetingSuggestionTimelineMetrics | null
}

export interface ElectronAPI {
  updateContentDimensions: (dimensions: {
    width: number
    height: number
  }) => Promise<void>
  getScreenshots: () => Promise<Array<{ path: string; preview: string }>>
  deleteScreenshot: (path: string) => Promise<{ success: boolean; error?: string }>
  onScreenshotTaken: (callback: (data: { path: string; preview: string }) => void) => () => void
  onSolutionsReady: (callback: (solutions: string) => void) => () => void
  onResetView: (callback: () => void) => () => void
  onSolutionStart: (callback: () => void) => () => void
  onDebugStart: (callback: () => void) => () => void
  onDebugSuccess: (callback: (data: any) => void) => () => void
  onSolutionError: (callback: (error: string) => void) => () => void
  onProcessingNoScreenshots: (callback: () => void) => () => void
  onProblemExtracted: (callback: (data: any) => void) => () => void
  onSolutionSuccess: (callback: (data: any) => void) => () => void
  onUnauthorized: (callback: () => void) => () => void
  onDebugError: (callback: (error: string) => void) => () => void
  onRealtimeTranscriptionUpdate: (callback: (data: { text: string; fullTranscript: string | null; metrics?: RealtimePartialMetrics }) => void) => () => void
  onRealtimeTranscriptionComplete: (callback: (data: { text: string; fullTranscript: string; metrics?: RealtimeCompleteMetrics }) => void) => () => void
  onRealtimeTranscriptionTimeout: (callback: (data: { text: string; fullTranscript: string; metrics?: RealtimeTimeoutMetrics }) => void) => () => void
  onRealtimeTranscriptionStatus: (callback: (data: { status: string; timestamp: number }) => void) => () => void
  takeScreenshot: () => Promise<void>
  moveWindowLeft: () => Promise<void>
  moveWindowRight: () => Promise<void>
  moveWindowUp: () => Promise<void>
  moveWindowDown: () => Promise<void>
  minimizeWindow: () => Promise<void>
  analyzeAudioFromBase64: (data: string, mimeType: string) => Promise<{ text: string; timestamp: number }>
  analyzeAudioFile: (path: string) => Promise<{ text: string; timestamp: number }>
  quitApp: () => Promise<void>
  // Transcription methods
  transcribeAudioFile: (audioPath: string, customPrompt?: string, filename?: string) => Promise<{ text: string; filePath: string; timestamp: number }>
  transcribeAudioFromBase64: (data: string, mimeType: string, customPrompt?: string, filename?: string) => Promise<{ text: string; filePath: string; timestamp: number }>
  saveTranscript: (transcriptText: string, filename?: string) => Promise<{ success: boolean; filePath?: string; error?: string }>
  // Real-time transcription methods
  startRealTimeTranscription: (filename?: string) => Promise<{ success: boolean; filePath?: string; error?: string }>
  processRealTimeAudioChunk: (data: string, mimeType: string, customPrompt?: string) => Promise<{ chunk: string; fullTranscript: string; timestamp: number }>
  stopRealTimeTranscription: () => Promise<{ success: boolean; filePath?: string; error?: string }>
  getRealTimeTranscript: () => Promise<{ success: boolean; transcript: string | null; error?: string }>
  isRealTimeTranscriptionActive: () => Promise<{ isActive: boolean }>
  generateMeetingSuggestion: (transcript: string, systemPrompt: string) => Promise<{ text: string; type: "response" | "question" | "negotiation"; metrics?: MeetingSuggestionMetrics }>
  invoke: (channel: string, ...args: any[]) => Promise<any>
  logToTerminal: (level: "info" | "warn" | "error" | "debug", ...args: any[]) => Promise<void>
}

declare global {
  interface Window {
    electronAPI: ElectronAPI
  }
}
