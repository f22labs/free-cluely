// ipcHandlers.ts

import { ipcMain, app } from "electron"
import { AppState } from "./main"
import { logger } from "./logger"

const metricsLoggingEnabled = process.env.RTSTT_METRICS_LOG === "1"

export function initializeIpcHandlers(appState: AppState): void {
  ipcMain.handle(
    "update-content-dimensions",
    async (event, { width, height }: { width: number; height: number }) => {
      if (width && height) {
        appState.setWindowDimensions(width, height)
      }
    }
  )

  ipcMain.handle("delete-screenshot", async (event, path: string) => {
    return appState.deleteScreenshot(path)
  })

  ipcMain.handle("take-screenshot", async () => {
    try {
      const screenshotPath = await appState.takeScreenshot()
      const preview = await appState.getImagePreview(screenshotPath)
      return { path: screenshotPath, preview }
    } catch (error) {
      logger.error("Error taking screenshot:", error)
      throw error
    }
  })

  ipcMain.handle("get-screenshots", async () => {
    logger.debug("[IPC] Current view:", appState.getView())
    try {
      let previews = []
      if (appState.getView() === "queue") {
        previews = await Promise.all(
          appState.getScreenshotQueue().map(async (path) => ({
            path,
            preview: await appState.getImagePreview(path)
          }))
        )
      } else {
        previews = await Promise.all(
          appState.getExtraScreenshotQueue().map(async (path) => ({
            path,
            preview: await appState.getImagePreview(path)
          }))
        )
      }
      previews.forEach((preview: any) => logger.debug("[IPC] Screenshot path:", preview.path))
      return previews
    } catch (error) {
      logger.error("Error getting screenshots:", error)
      throw error
    }
  })

  ipcMain.handle("toggle-window", async () => {
    appState.toggleMainWindow()
  })

  ipcMain.handle("reset-queues", async () => {
    try {
      appState.clearQueues()
      logger.info("Screenshot queues have been cleared.")
      return { success: true }
    } catch (error: any) {
      logger.error("Error resetting queues:", error)
      return { success: false, error: error.message }
    }
  })

  // IPC handler for analyzing audio from base64 data
  ipcMain.handle("analyze-audio-base64", async (event, data: string, mimeType: string) => {
    try {
      const result = await appState.processingHelper.processAudioBase64(data, mimeType)
      return result
    } catch (error: any) {
      logger.error("Error in analyze-audio-base64 handler:", error)
      throw error
    }
  })

  // IPC handler for analyzing audio from file path
  ipcMain.handle("analyze-audio-file", async (event, path: string) => {
    try {
      const result = await appState.processingHelper.processAudioFile(path)
      return result
    } catch (error: any) {
      logger.error("Error in analyze-audio-file handler:", error)
      throw error
    }
  })

  // IPC handler for analyzing image from file path
  ipcMain.handle("analyze-image-file", async (event, path: string) => {
    try {
      const result = await appState.processingHelper.getLLMHelper().analyzeImageFile(path)
      return result
    } catch (error: any) {
      logger.error("Error in analyze-image-file handler:", error)
      throw error
    }
  })

  ipcMain.handle("gemini-chat", async (event, message: string) => {
    try {
      const result = await appState.processingHelper.getLLMHelper().chatWithGemini(message);
      return result;
    } catch (error: any) {
      logger.error("Error in gemini-chat handler:", error);
      throw error;
    }
  });

  ipcMain.handle("quit-app", () => {
    app.quit()
  })

  // Window movement handlers
  ipcMain.handle("move-window-left", async () => {
    appState.moveWindowLeft()
  })

  ipcMain.handle("move-window-right", async () => {
    appState.moveWindowRight()
  })

  ipcMain.handle("move-window-up", async () => {
    appState.moveWindowUp()
  })

  ipcMain.handle("move-window-down", async () => {
    appState.moveWindowDown()
  })

  ipcMain.handle("center-and-show-window", async () => {
    appState.centerAndShowWindow()
  })

  // LLM Model Management Handlers
  ipcMain.handle("get-current-llm-config", async () => {
    try {
      const llmHelper = appState.processingHelper.getLLMHelper();
      return {
        provider: llmHelper.getCurrentProvider(),
        model: llmHelper.getCurrentModel(),
        isOllama: llmHelper.isUsingOllama()
      };
    } catch (error: any) {
      logger.error("Error getting current LLM config:", error);
      throw error;
    }
  });

  ipcMain.handle("get-available-ollama-models", async () => {
    try {
      const llmHelper = appState.processingHelper.getLLMHelper();
      const models = await llmHelper.getOllamaModels();
      return models;
    } catch (error: any) {
      logger.error("Error getting Ollama models:", error);
      throw error;
    }
  });

  ipcMain.handle("switch-to-ollama", async (_, model?: string, url?: string) => {
    try {
      const llmHelper = appState.processingHelper.getLLMHelper();
      await llmHelper.switchToOllama(model, url);
      return { success: true };
    } catch (error: any) {
      logger.error("Error switching to Ollama:", error);
      return { success: false, error: error.message };
    }
  });

  ipcMain.handle("switch-to-gemini", async (_, apiKey?: string) => {
    try {
      const llmHelper = appState.processingHelper.getLLMHelper();
      await llmHelper.switchToGemini(apiKey);
      return { success: true };
    } catch (error: any) {
      logger.error("Error switching to Gemini:", error);
      return { success: false, error: error.message };
    }
  });

  ipcMain.handle("test-llm-connection", async () => {
    try {
      const llmHelper = appState.processingHelper.getLLMHelper();
      const result = await llmHelper.testConnection();
      return result;
    } catch (error: any) {
      logger.error("Error testing LLM connection:", error);
      return { success: false, error: error.message };
    }
  });

  // IPC handler for transcribing audio file with custom prompt
  ipcMain.handle("transcribe-audio-file", async (event, audioPath: string, customPrompt?: string, filename?: string) => {
    try {
      const result = await appState.processingHelper.transcribeAndSaveAudioFile(audioPath, customPrompt, filename)
      return result
    } catch (error: any) {
      logger.error("Error in transcribe-audio-file handler:", error)
      throw error
    }
  })

  // IPC handler for transcribing audio from base64 with custom prompt
  ipcMain.handle("transcribe-audio-base64", async (event, data: string, mimeType: string, customPrompt?: string, filename?: string) => {
    try {
      const result = await appState.processingHelper.transcribeAndSaveAudioFromBase64(data, mimeType, customPrompt, filename)
      return result
    } catch (error: any) {
      logger.error("Error in transcribe-audio-base64 handler:", error)
      throw error
    }
  })

  // IPC handler for saving transcript text to file
  ipcMain.handle("save-transcript", async (event, transcriptText: string, filename?: string) => {
    try {
      const filePath = await appState.processingHelper.saveTranscriptToFile(transcriptText, filename)
      return { success: true, filePath }
    } catch (error: any) {
      logger.error("Error in save-transcript handler:", error)
      return { success: false, error: error.message }
    }
  })

  // Real-time transcription handlers
  ipcMain.handle("start-realtime-transcription", async (event, filename?: string) => {
    try {
      const filePath = await appState.processingHelper.startRealTimeTranscription(filename)
      return { success: true, filePath }
    } catch (error: any) {
      logger.error("Error in start-realtime-transcription handler:", error)
      return { success: false, error: error.message }
    }
  })

  ipcMain.handle("process-realtime-audio-chunk", async (event, data: string, mimeType: string, customPrompt?: string) => {
    try {
      const result = await appState.processingHelper.processRealTimeAudioChunk(data, mimeType, customPrompt)
      return result
    } catch (error: any) {
      logger.error("Error in process-realtime-audio-chunk handler:", error)
      throw error
    }
  })

  ipcMain.handle("stop-realtime-transcription", async () => {
    try {
      const filePath = await appState.processingHelper.stopRealTimeTranscription()
      return { success: true, filePath }
    } catch (error: any) {
      logger.error("Error in stop-realtime-transcription handler:", error)
      return { success: false, error: error.message }
    }
  })

  ipcMain.handle("get-realtime-transcript", async () => {
    try {
      const transcript = await appState.processingHelper.getRealTimeTranscript()
      return { success: true, transcript }
    } catch (error: any) {
      logger.error("Error in get-realtime-transcript handler:", error)
      return { success: false, error: error.message, transcript: null }
    }
  })

  ipcMain.handle("is-realtime-transcription-active", async () => {
    try {
      const isActive = appState.processingHelper.isRealTimeTranscriptionActive()
      return { isActive }
    } catch (error: any) {
      logger.error("Error in is-realtime-transcription-active handler:", error)
      return { isActive: false }
    }
  })

  // Meeting Assistant handlers
  ipcMain.handle("generate-meeting-suggestion", async (event, transcript: string, systemPrompt: string) => {
    try {
      logger.info("[IPC] generate-meeting-suggestion called with transcript length:", transcript.length)

      const { metrics: transcriptionMetrics, requestEpochMs } = appState.processingHelper.noteSuggestionRequest()
      if (transcriptionMetrics) {
        const sinceElectron = requestEpochMs - transcriptionMetrics.electronReceivedAt
        const sincePython = transcriptionMetrics.pythonCompletionEpochMs != null
          ? requestEpochMs - transcriptionMetrics.pythonCompletionEpochMs
          : null
        if (metricsLoggingEnabled) {
          logger.info(
            `[Metrics][Suggestion] Time since transcription: renderer_request=${sinceElectron}ms python_reference=${sincePython ?? "n/a"}ms (iteration=${transcriptionMetrics.iteration ?? "?"})`
          )
        }
      } else {
        if (metricsLoggingEnabled) {
          logger.info("[Metrics][Suggestion] No transcription metrics snapshot available for this suggestion request.")
        }
      }

      const llmHelper = appState.processingHelper.getLLMHelper()
      const llmStartEpochMs = Date.now()
      const suggestion = await llmHelper.generateMeetingSuggestion(transcript, systemPrompt)
      const llmCompletedEpochMs = Date.now()
      const llmDurationMs = llmCompletedEpochMs - llmStartEpochMs
      appState.processingHelper.noteSuggestionResponse(llmDurationMs)

      logger.info("[IPC] Generated suggestion:", suggestion)
      if (metricsLoggingEnabled) {
        logger.info(
          `[Metrics][Suggestion] LLM duration: ${llmDurationMs}ms (round trip ${llmCompletedEpochMs - requestEpochMs}ms)`
        )
      }

      const transcriptionTimeline = transcriptionMetrics ? {
        iteration: transcriptionMetrics.iteration,
        python_latency_ms: transcriptionMetrics.pythonLatencyMs,
        python_first_partial_latency_ms: transcriptionMetrics.firstPartialLatencyMs,
        python_to_electron_ms: transcriptionMetrics.pythonToElectronMs,
        electron_received_epoch_ms: transcriptionMetrics.electronReceivedAt,
        electron_emit_timestamp: transcriptionMetrics.electronEmitTimestamp,
        renderer_request_epoch_ms: requestEpochMs,
        transcription_to_request_ms: requestEpochMs - transcriptionMetrics.electronReceivedAt,
        transcription_to_request_python_ms: transcriptionMetrics.pythonCompletionEpochMs != null
          ? requestEpochMs - transcriptionMetrics.pythonCompletionEpochMs
          : null,
        fallback_used: transcriptionMetrics.fallbackUsed,
        partial_update_count: transcriptionMetrics.partialUpdateCount
      } : null

      const responseMetrics = {
        ...(suggestion.metrics ?? {}),
        llm_duration_ms: suggestion.metrics?.llm_duration_ms ?? llmDurationMs,
        llm_started_at: suggestion.metrics?.llm_started_at ?? new Date(llmStartEpochMs).toISOString(),
        llm_completed_at: suggestion.metrics?.llm_completed_at ?? new Date(llmCompletedEpochMs).toISOString(),
        llm_round_trip_duration_ms: llmCompletedEpochMs - requestEpochMs,
        renderer_request_epoch_ms: requestEpochMs,
        renderer_response_epoch_ms: llmCompletedEpochMs,
        transcription_timeline: transcriptionTimeline
      }

      return {
        ...suggestion,
        metrics: responseMetrics
      }
    } catch (error: any) {
      logger.error("Error in generate-meeting-suggestion handler:", error)
      throw error
    }
  })
}
