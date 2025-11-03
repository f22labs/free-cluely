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
  onRealtimeTranscriptionUpdate: (callback: (data: { text: string; fullTranscript: string | null }) => void) => () => void
  onRealtimeTranscriptionComplete: (callback: (data: { text: string; fullTranscript: string }) => void) => () => void
  takeScreenshot: () => Promise<void>
  moveWindowLeft: () => Promise<void>
  moveWindowRight: () => Promise<void>
  moveWindowUp: () => Promise<void>
  moveWindowDown: () => Promise<void>
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
  invoke: (channel: string, ...args: any[]) => Promise<any>
}

declare global {
  interface Window {
    electronAPI: ElectronAPI
  }
} 