// ipcHandlers.ts

import { ipcMain, app } from "electron"
import { AppState } from "./main"

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
      console.error("Error taking screenshot:", error)
      throw error
    }
  })

  ipcMain.handle("get-screenshots", async () => {
    console.log({ view: appState.getView() })
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
      previews.forEach((preview: any) => console.log(preview.path))
      return previews
    } catch (error) {
      console.error("Error getting screenshots:", error)
      throw error
    }
  })

  ipcMain.handle("toggle-window", async () => {
    appState.toggleMainWindow()
  })

  ipcMain.handle("reset-queues", async () => {
    try {
      appState.clearQueues()
      console.log("Screenshot queues have been cleared.")
      return { success: true }
    } catch (error: any) {
      console.error("Error resetting queues:", error)
      return { success: false, error: error.message }
    }
  })

  // IPC handler for analyzing audio from base64 data
  ipcMain.handle("analyze-audio-base64", async (event, data: string, mimeType: string) => {
    try {
      const result = await appState.processingHelper.processAudioBase64(data, mimeType)
      return result
    } catch (error: any) {
      console.error("Error in analyze-audio-base64 handler:", error)
      throw error
    }
  })

  // IPC handler for analyzing audio from file path
  ipcMain.handle("analyze-audio-file", async (event, path: string) => {
    try {
      const result = await appState.processingHelper.processAudioFile(path)
      return result
    } catch (error: any) {
      console.error("Error in analyze-audio-file handler:", error)
      throw error
    }
  })

  // IPC handler for analyzing image from file path
  ipcMain.handle("analyze-image-file", async (event, path: string) => {
    try {
      const result = await appState.processingHelper.getLLMHelper().analyzeImageFile(path)
      return result
    } catch (error: any) {
      console.error("Error in analyze-image-file handler:", error)
      throw error
    }
  })

  ipcMain.handle("gemini-chat", async (event, message: string) => {
    try {
      const result = await appState.processingHelper.getLLMHelper().chatWithGemini(message);
      return result;
    } catch (error: any) {
      console.error("Error in gemini-chat handler:", error);
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
      console.error("Error getting current LLM config:", error);
      throw error;
    }
  });

  ipcMain.handle("get-available-ollama-models", async () => {
    try {
      const llmHelper = appState.processingHelper.getLLMHelper();
      const models = await llmHelper.getOllamaModels();
      return models;
    } catch (error: any) {
      console.error("Error getting Ollama models:", error);
      throw error;
    }
  });

  ipcMain.handle("switch-to-ollama", async (_, model?: string, url?: string) => {
    try {
      const llmHelper = appState.processingHelper.getLLMHelper();
      await llmHelper.switchToOllama(model, url);
      return { success: true };
    } catch (error: any) {
      console.error("Error switching to Ollama:", error);
      return { success: false, error: error.message };
    }
  });

  ipcMain.handle("switch-to-gemini", async (_, apiKey?: string) => {
    try {
      const llmHelper = appState.processingHelper.getLLMHelper();
      await llmHelper.switchToGemini(apiKey);
      return { success: true };
    } catch (error: any) {
      console.error("Error switching to Gemini:", error);
      return { success: false, error: error.message };
    }
  });

  ipcMain.handle("test-llm-connection", async () => {
    try {
      const llmHelper = appState.processingHelper.getLLMHelper();
      const result = await llmHelper.testConnection();
      return result;
    } catch (error: any) {
      console.error("Error testing LLM connection:", error);
      return { success: false, error: error.message };
    }
  });

  // IPC handler for transcribing audio file with custom prompt
  ipcMain.handle("transcribe-audio-file", async (event, audioPath: string, customPrompt?: string, filename?: string) => {
    try {
      const result = await appState.processingHelper.transcribeAndSaveAudioFile(audioPath, customPrompt, filename)
      return result
    } catch (error: any) {
      console.error("Error in transcribe-audio-file handler:", error)
      throw error
    }
  })

  // IPC handler for transcribing audio from base64 with custom prompt
  ipcMain.handle("transcribe-audio-base64", async (event, data: string, mimeType: string, customPrompt?: string, filename?: string) => {
    try {
      const result = await appState.processingHelper.transcribeAndSaveAudioFromBase64(data, mimeType, customPrompt, filename)
      return result
    } catch (error: any) {
      console.error("Error in transcribe-audio-base64 handler:", error)
      throw error
    }
  })

  // IPC handler for saving transcript text to file
  ipcMain.handle("save-transcript", async (event, transcriptText: string, filename?: string) => {
    try {
      const filePath = await appState.processingHelper.saveTranscriptToFile(transcriptText, filename)
      return { success: true, filePath }
    } catch (error: any) {
      console.error("Error in save-transcript handler:", error)
      return { success: false, error: error.message }
    }
  })

  // Real-time transcription handlers
  ipcMain.handle("start-realtime-transcription", async (event, filename?: string) => {
    try {
      const filePath = await appState.processingHelper.startRealTimeTranscription(filename)
      return { success: true, filePath }
    } catch (error: any) {
      console.error("Error in start-realtime-transcription handler:", error)
      return { success: false, error: error.message }
    }
  })

  ipcMain.handle("process-realtime-audio-chunk", async (event, data: string, mimeType: string, customPrompt?: string) => {
    try {
      const result = await appState.processingHelper.processRealTimeAudioChunk(data, mimeType, customPrompt)
      return result
    } catch (error: any) {
      console.error("Error in process-realtime-audio-chunk handler:", error)
      throw error
    }
  })

  ipcMain.handle("stop-realtime-transcription", async () => {
    try {
      const filePath = await appState.processingHelper.stopRealTimeTranscription()
      return { success: true, filePath }
    } catch (error: any) {
      console.error("Error in stop-realtime-transcription handler:", error)
      return { success: false, error: error.message }
    }
  })

  ipcMain.handle("get-realtime-transcript", async () => {
    try {
      const transcript = await appState.processingHelper.getRealTimeTranscript()
      return { success: true, transcript }
    } catch (error: any) {
      console.error("Error in get-realtime-transcript handler:", error)
      return { success: false, error: error.message, transcript: null }
    }
  })

  ipcMain.handle("is-realtime-transcription-active", async () => {
    try {
      const isActive = appState.processingHelper.isRealTimeTranscriptionActive()
      return { isActive }
    } catch (error: any) {
      console.error("Error in is-realtime-transcription-active handler:", error)
      return { isActive: false }
    }
  })
}
