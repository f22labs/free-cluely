// ProcessingHelper.ts

import { AppState } from "./main"
import { LLMHelper } from "./LLMHelper"
import dotenv from "dotenv"
import path from "node:path"
import fs from "node:fs"
import { app } from "electron"
import { spawn, ChildProcess } from "child_process"

// Transcripts directory - saved to project root for easy access
// Note: When compiled, __dirname will be dist-electron, so this resolves to project root/transcripts
const TRANSCRIPTS_DIR = path.resolve(path.join(__dirname, "..", "transcripts"))
console.log(`[ProcessingHelper] Transcripts directory will be: ${TRANSCRIPTS_DIR}`)

dotenv.config()

const isDev = process.env.NODE_ENV === "development"
const isDevTest = process.env.IS_DEV_TEST === "true"
const MOCK_API_WAIT_TIME = Number(process.env.MOCK_API_WAIT_TIME) || 500

export class ProcessingHelper {
  private appState: AppState
  private llmHelper: LLMHelper
  private currentProcessingAbortController: AbortController | null = null
  private currentExtraProcessingAbortController: AbortController | null = null
  private realTimeTranscriptSession: {
    filePath: string
    filename: string
    startTime: number
  } | null = null
  private realtimeSTTProcess: ChildProcess | null = null

  constructor(appState: AppState) {
    this.appState = appState
    
    // Check if user wants to use Ollama
    const useOllama = process.env.USE_OLLAMA === "true"
    const ollamaModel = process.env.OLLAMA_MODEL // Don't set default here, let LLMHelper auto-detect
    const ollamaUrl = process.env.OLLAMA_URL || "http://localhost:11434"
    
    if (useOllama) {
      console.log("[ProcessingHelper] Initializing with Ollama")
      this.llmHelper = new LLMHelper(undefined, true, ollamaModel, ollamaUrl)
    } else {
      const apiKey = process.env.GEMINI_API_KEY
      if (!apiKey) {
        throw new Error("GEMINI_API_KEY not found in environment variables. Set GEMINI_API_KEY or enable Ollama with USE_OLLAMA=true")
      }
      console.log("[ProcessingHelper] Initializing with Gemini")
      this.llmHelper = new LLMHelper(apiKey, false)
    }
  }

  public async processScreenshots(): Promise<void> {
    const mainWindow = this.appState.getMainWindow()
    if (!mainWindow) return

    const view = this.appState.getView()

    if (view === "queue") {
      const screenshotQueue = this.appState.getScreenshotHelper().getScreenshotQueue()
      if (screenshotQueue.length === 0) {
        mainWindow.webContents.send(this.appState.PROCESSING_EVENTS.NO_SCREENSHOTS)
        return
      }

      // Check if last screenshot is an audio file
      const allPaths = this.appState.getScreenshotHelper().getScreenshotQueue();
      const lastPath = allPaths[allPaths.length - 1];
      if (lastPath.endsWith('.mp3') || lastPath.endsWith('.wav')) {
        mainWindow.webContents.send(this.appState.PROCESSING_EVENTS.INITIAL_START);
        this.appState.setView('solutions');
        try {
          const audioResult = await this.llmHelper.analyzeAudioFile(lastPath);
          mainWindow.webContents.send(this.appState.PROCESSING_EVENTS.PROBLEM_EXTRACTED, audioResult);
          this.appState.setProblemInfo({ problem_statement: audioResult.text, input_format: {}, output_format: {}, constraints: [], test_cases: [] });
          return;
        } catch (err: any) {
          console.error('Audio processing error:', err);
          mainWindow.webContents.send(this.appState.PROCESSING_EVENTS.INITIAL_SOLUTION_ERROR, err.message);
          return;
        }
      }

      // NEW: Handle screenshot as plain text (like audio)
      mainWindow.webContents.send(this.appState.PROCESSING_EVENTS.INITIAL_START)
      this.appState.setView("solutions")
      this.currentProcessingAbortController = new AbortController()
      try {
        const imageResult = await this.llmHelper.analyzeImageFile(lastPath);
        const problemInfo = {
          problem_statement: imageResult.text,
          input_format: { description: "Generated from screenshot", parameters: [] as any[] },
          output_format: { description: "Generated from screenshot", type: "string", subtype: "text" },
          complexity: { time: "N/A", space: "N/A" },
          test_cases: [] as any[],
          validation_type: "manual",
          difficulty: "custom"
        };
        mainWindow.webContents.send(this.appState.PROCESSING_EVENTS.PROBLEM_EXTRACTED, problemInfo);
        this.appState.setProblemInfo(problemInfo);
      } catch (error: any) {
        console.error("Image processing error:", error)
        mainWindow.webContents.send(this.appState.PROCESSING_EVENTS.INITIAL_SOLUTION_ERROR, error.message)
      } finally {
        this.currentProcessingAbortController = null
      }
      return;
    } else {
      // Debug mode
      const extraScreenshotQueue = this.appState.getScreenshotHelper().getExtraScreenshotQueue()
      if (extraScreenshotQueue.length === 0) {
        console.log("No extra screenshots to process")
        mainWindow.webContents.send(this.appState.PROCESSING_EVENTS.NO_SCREENSHOTS)
        return
      }

      mainWindow.webContents.send(this.appState.PROCESSING_EVENTS.DEBUG_START)
      this.currentExtraProcessingAbortController = new AbortController()

      try {
        // Get problem info and current solution
        const problemInfo = this.appState.getProblemInfo()
        if (!problemInfo) {
          throw new Error("No problem info available")
        }

        // Get current solution from state
        const currentSolution = await this.llmHelper.generateSolution(problemInfo)
        const currentCode = currentSolution.solution.code

        // Debug the solution using vision model
        const debugResult = await this.llmHelper.debugSolutionWithImages(
          problemInfo,
          currentCode,
          extraScreenshotQueue
        )

        this.appState.setHasDebugged(true)
        mainWindow.webContents.send(
          this.appState.PROCESSING_EVENTS.DEBUG_SUCCESS,
          debugResult
        )

      } catch (error: any) {
        console.error("Debug processing error:", error)
        mainWindow.webContents.send(
          this.appState.PROCESSING_EVENTS.DEBUG_ERROR,
          error.message
        )
      } finally {
        this.currentExtraProcessingAbortController = null
      }
    }
  }

  public cancelOngoingRequests(): void {
    if (this.currentProcessingAbortController) {
      this.currentProcessingAbortController.abort()
      this.currentProcessingAbortController = null
    }

    if (this.currentExtraProcessingAbortController) {
      this.currentExtraProcessingAbortController.abort()
      this.currentExtraProcessingAbortController = null
    }

    this.appState.setHasDebugged(false)
  }

  public async processAudioBase64(data: string, mimeType: string) {
    // Analyze audio using LLMHelper (for UI display)
    const result = await this.llmHelper.analyzeAudioFromBase64(data, mimeType);
    
    // Save transcript: if there's an active real-time session, append to it; otherwise create a new file
    try {
      const transcript = await this.llmHelper.transcribeAudioFromBase64(data, mimeType);
      
      // Check if there's an active real-time transcription session
      if (this.isRealTimeTranscriptionActive()) {
        // Append to the active session file
        await this.appendToRealTimeTranscript(transcript.text);
        console.log(`[ProcessingHelper] Audio transcript appended to active session`);
      } else {
        // No active session - create a new file for this single audio analysis
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `transcript_${timestamp}.txt`;
        const filePath = await this.saveTranscriptToFile(transcript.text, filename);
        console.log(`[ProcessingHelper] Audio transcript saved to: ${filePath}`);
      }
    } catch (error) {
      console.error("[ProcessingHelper] Error saving audio transcript:", error);
      // Don't throw - we still want to return the result even if saving fails
    }
    
    return result;
  }

  // Add audio file processing method
  public async processAudioFile(filePath: string) {
    // Analyze audio file using LLMHelper (for UI display)
    const result = await this.llmHelper.analyzeAudioFile(filePath);
    
    // Also save ONLY the transcript (word-for-word transcription) to a file automatically
    try {
      const transcript = await this.llmHelper.transcribeAudioFile(filePath);
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const filename = `transcript_${timestamp}.txt`;
      const savedFilePath = await this.saveTranscriptToFile(transcript.text, filename);
      console.log(`[ProcessingHelper] Audio file transcript saved to: ${savedFilePath}`);
    } catch (error) {
      console.error("[ProcessingHelper] Error saving audio file transcript:", error);
      // Don't throw - we still want to return the result even if saving fails
    }
    
    return result;
  }

  public getLLMHelper() {
    return this.llmHelper;
  }

  /**
   * Save transcript to a local file
   * @param transcriptText The transcript text to save
   * @param filename Optional custom filename. If not provided, uses timestamp-based name.
   * @returns Path to the saved file
   */
  public async saveTranscriptToFile(transcriptText: string, filename?: string): Promise<string> {
    try {
      // Use project root transcripts directory for easy access
      const transcriptsDir = TRANSCRIPTS_DIR;
      
      // Create transcripts directory if it doesn't exist
      if (!fs.existsSync(transcriptsDir)) {
        fs.mkdirSync(transcriptsDir, { recursive: true });
      }
      
      // Generate filename if not provided
      const transcriptFilename = filename || `transcript_${Date.now()}.txt`;
      const filePath = path.join(transcriptsDir, transcriptFilename);
      
      // Write transcript to file
      await fs.promises.writeFile(filePath, transcriptText, "utf-8");
      
      console.log(`[ProcessingHelper] Transcript saved to: ${filePath}`);
      return filePath;
    } catch (error) {
      console.error("[ProcessingHelper] Error saving transcript:", error);
      throw error;
    }
  }

  /**
   * Transcribe audio file with custom prompt and save to local file
   * @param audioPath Path to audio file
   * @param customPrompt Optional custom transcription prompt
   * @param filename Optional filename for saved transcript
   * @returns Object containing transcript text, file path, and timestamp
   */
  public async transcribeAndSaveAudioFile(
    audioPath: string, 
    customPrompt?: string, 
    filename?: string
  ): Promise<{ text: string; filePath: string; timestamp: number }> {
    try {
      // Get transcript using LLM
      const transcript = await this.llmHelper.transcribeAudioFile(audioPath, customPrompt);
      
      // Save to file
      const filePath = await this.saveTranscriptToFile(transcript.text, filename);
      
      return {
        text: transcript.text,
        filePath,
        timestamp: transcript.timestamp
      };
    } catch (error) {
      console.error("[ProcessingHelper] Error in transcribeAndSaveAudioFile:", error);
      throw error;
    }
  }

  /**
   * Transcribe audio from base64 with custom prompt and save to local file
   * @param data Base64 encoded audio data
   * @param mimeType MIME type of the audio
   * @param customPrompt Optional custom transcription prompt
   * @param filename Optional filename for saved transcript
   * @returns Object containing transcript text, file path, and timestamp
   */
  public async transcribeAndSaveAudioFromBase64(
    data: string,
    mimeType: string,
    customPrompt?: string,
    filename?: string
  ): Promise<{ text: string; filePath: string; timestamp: number }> {
    try {
      // Get transcript using LLM
      const transcript = await this.llmHelper.transcribeAudioFromBase64(data, mimeType, customPrompt);
      
      // Save to file
      const filePath = await this.saveTranscriptToFile(transcript.text, filename);
      
      return {
        text: transcript.text,
        filePath,
        timestamp: transcript.timestamp
      };
    } catch (error) {
      console.error("[ProcessingHelper] Error in transcribeAndSaveAudioFromBase64:", error);
      throw error;
    }
  }

  /**
   * Start a real-time transcription session using RealtimeSTT Python service
   * Creates a new transcript file and spawns Python process for fast transcription
   * @param filename Optional custom filename
   * @returns File path of the transcript file
   */
  public async startRealTimeTranscription(filename?: string): Promise<string> {
    try {
      // Stop any existing process
      if (this.realtimeSTTProcess) {
        await this.stopRealTimeTranscription();
      }

      // Use project root transcripts directory for easy access
      const transcriptsDir = TRANSCRIPTS_DIR;
      
      // Create transcripts directory if it doesn't exist
      if (!fs.existsSync(transcriptsDir)) {
        fs.mkdirSync(transcriptsDir, { recursive: true });
      }
      
      // Generate filename if not provided
      const transcriptFilename = filename || `transcript_${Date.now()}.txt`;
      const filePath = path.join(transcriptsDir, transcriptFilename);
      
      // Store session info
      this.realTimeTranscriptSession = {
        filePath,
        filename: transcriptFilename,
        startTime: Date.now()
      };
      
      // Find Python script path
      const pythonScriptPath = path.join(__dirname, "..", "realtime_stt_service.py");
      
      // Spawn Python process for RealtimeSTT
      // Using small.en for real-time (good balance of speed and accuracy)
      // Using large-v3 for final transcription (latest and most accurate model)
      const pythonProcess = spawn("python3", [
        pythonScriptPath,
        "--transcript-file", filePath,
        "--model", "large-v2",  // Final accuracy model
        "--realtime-model", "small.en",  // Good balance - medium.en may be too slow for continuous real-time
        "--language", "en"
      ], {
        stdio: ["pipe", "pipe", "pipe"],
        cwd: path.join(__dirname, "..")
      });
      
      this.realtimeSTTProcess = pythonProcess;
      
      // Handle stdout (JSON messages from Python)
      let stdoutBuffer = "";
      pythonProcess.stdout?.on("data", (data: Buffer) => {
        stdoutBuffer += data.toString();
        const lines = stdoutBuffer.split("\n");
        stdoutBuffer = lines.pop() || ""; // Keep incomplete line
        
        for (const line of lines) {
          if (line.trim()) {
            try {
              const message = JSON.parse(line);
              this.handleRealtimeSTTMessage(message);
            } catch (e) {
              // Ignore non-JSON lines
            }
          }
        }
      });
      
      // Handle stderr (errors/logs/debug output)
      let stderrBuffer = "";
      pythonProcess.stderr?.on("data", (data: Buffer) => {
        stderrBuffer += data.toString();
        const lines = stderrBuffer.split("\n");
        stderrBuffer = lines.pop() || ""; // Keep incomplete line
        
        for (const line of lines) {
          if (line.trim()) {
            try {
              // Try to parse as JSON first
              const message = JSON.parse(line);
              if (message.type === "error") {
                console.error("[RealtimeSTT] Error:", message.error);
              } else if (message.type === "status") {
                console.log("[RealtimeSTT] Status:", message.status);
              }
            } catch (e) {
              // Not JSON - print all stderr output (includes debug messages)
              console.log("[RealtimeSTT DEBUG]", line);
            }
          }
        }
      });
      
      // Handle process exit
      pythonProcess.on("exit", (code) => {
        console.log(`[RealtimeSTT] Process exited with code ${code}`);
        this.realtimeSTTProcess = null;
      });
      
      pythonProcess.on("error", (error) => {
        console.error("[RealtimeSTT] Process error:", error);
        this.realtimeSTTProcess = null;
      });
      
      console.log(`[ProcessingHelper] Real-time transcription session started with RealtimeSTT: ${filePath}`);
      return filePath;
    } catch (error) {
      console.error("[ProcessingHelper] Error starting real-time transcription:", error);
      throw error;
    }
  }

  /**
   * Handle messages from RealtimeSTT Python service
   */
  private handleRealtimeSTTMessage(message: any): void {
    const mainWindow = this.appState.getMainWindow();
    if (!mainWindow) return;

    switch (message.type) {
      case "realtime_update":
        // Real-time update - notify frontend (don't append to file, prevents duplicates)
        if (message.text && this.realTimeTranscriptSession) {
          // Send update to frontend with full transcript context
          mainWindow.webContents.send("realtime-transcription-update", {
            text: message.text,  // Current partial sentence
            fullTranscript: message.fullTranscript || message.text  // Complete + partial
          });
        }
        break;
        
      case "transcription_complete":
        // Completed transcription - update file and notify
        if (message.text && this.realTimeTranscriptSession) {
          this.appendToRealTimeTranscript(message.text).catch(err => {
            console.error("[ProcessingHelper] Error appending completed transcription:", err);
          });
          
          // Send complete transcription to frontend
          mainWindow.webContents.send("realtime-transcription-complete", {
            text: message.text,
            fullTranscript: message.full_transcript || ""
          });
        }
        break;
        
      case "status":
        console.log(`[RealtimeSTT] Status: ${message.status}`);
        break;
        
      case "error":
        console.error(`[RealtimeSTT] Error: ${message.error}`);
        break;
    }
  }

  /**
   * Append text to the current real-time transcript file
   * @param text Text to append
   * @returns Updated file content
   */
  public async appendToRealTimeTranscript(text: string): Promise<string> {
    if (!this.realTimeTranscriptSession) {
      throw new Error("No active real-time transcription session");
    }

    try {
      // Append text with newline
      await fs.promises.appendFile(this.realTimeTranscriptSession.filePath, text + "\n", "utf-8");
      
      // Read and return the full content
      const content = await fs.promises.readFile(this.realTimeTranscriptSession.filePath, "utf-8");
      return content;
    } catch (error) {
      console.error("[ProcessingHelper] Error appending to real-time transcript:", error);
      throw error;
    }
  }

  /**
   * Process an audio chunk and append the transcription to the real-time transcript
   * @param data Base64 encoded audio data
   * @param mimeType MIME type of the audio
   * @param customPrompt Optional custom transcription prompt
   * @returns Object containing the new transcript chunk and full file content
   */
  public async processRealTimeAudioChunk(
    data: string,
    mimeType: string,
    customPrompt?: string
  ): Promise<{ chunk: string; fullTranscript: string; timestamp: number }> {
    if (!this.realTimeTranscriptSession) {
      throw new Error("No active real-time transcription session. Call startRealTimeTranscription first.");
    }

    // Validate audio data
    if (!data || data.length < 100) {
      console.warn("[ProcessingHelper] Audio chunk too small, skipping transcription");
      const currentTranscript = await this.getRealTimeTranscript();
      return {
        chunk: "",
        fullTranscript: currentTranscript || "",
        timestamp: Date.now()
      };
    }

    try {
      // Normalize MIME type - Gemini may not accept all webm variants
      let normalizedMimeType = mimeType;
      if (mimeType.includes('webm')) {
        normalizedMimeType = 'audio/webm';
      } else if (mimeType.includes('mp3')) {
        normalizedMimeType = 'audio/mp3';
      } else if (mimeType.includes('wav')) {
        normalizedMimeType = 'audio/wav';
      } else if (!mimeType || mimeType === 'application/octet-stream') {
        // Default to webm if unknown
        normalizedMimeType = 'audio/webm';
      }

      // Transcribe the audio chunk
      const transcript = await this.llmHelper.transcribeAudioFromBase64(data, normalizedMimeType, customPrompt);
      
      // Only append if we got actual transcript text
      if (transcript.text && transcript.text.trim().length > 0) {
        const fullTranscript = await this.appendToRealTimeTranscript(transcript.text);
        return {
          chunk: transcript.text,
          fullTranscript,
          timestamp: transcript.timestamp
        };
      } else {
        // Empty transcript - return current state
        const currentTranscript = await this.getRealTimeTranscript();
        return {
          chunk: "",
          fullTranscript: currentTranscript || "",
          timestamp: Date.now()
        };
      }
    } catch (error: any) {
      // Log the error but don't throw - we want to continue processing other chunks
      console.error("[ProcessingHelper] Error processing real-time audio chunk:", error.message || error);
      
      // Check if it's a validation error - these are likely due to invalid/incomplete chunks
      if (error.message && (error.message.includes('400') || error.message.includes('invalid argument'))) {
        console.warn("[ProcessingHelper] Skipping invalid audio chunk - likely too small or incomplete");
        
        // Return current transcript state without appending
        const currentTranscript = await this.getRealTimeTranscript();
        return {
          chunk: "",
          fullTranscript: currentTranscript || "",
          timestamp: Date.now()
        };
      }
      
      // For other errors, still throw to let the caller handle it
      throw error;
    }
  }

  /**
   * Stop the real-time transcription session and return final file path
   * @returns File path of the transcript
   */
  public async stopRealTimeTranscription(): Promise<string> {
    if (!this.realTimeTranscriptSession) {
      throw new Error("No active real-time transcription session");
    }

    try {
      const filePath = this.realTimeTranscriptSession.filePath;
      
      // Stop Python process if running
      if (this.realtimeSTTProcess) {
        try {
          // Send stop command
          this.realtimeSTTProcess.stdin?.write(JSON.stringify({ action: "stop" }) + "\n");
          this.realtimeSTTProcess.stdin?.end();
          
          // Give it a moment to finalize, then kill if still running
          setTimeout(() => {
            if (this.realtimeSTTProcess && !this.realtimeSTTProcess.killed) {
              this.realtimeSTTProcess.kill();
            }
          }, 2000);
        } catch (err) {
          console.error("[ProcessingHelper] Error stopping Python process:", err);
          if (this.realtimeSTTProcess) {
            this.realtimeSTTProcess.kill();
          }
        }
        
        this.realtimeSTTProcess = null;
      }
      
      const duration = Date.now() - this.realTimeTranscriptSession.startTime;
      
      // Read final content and ensure footer is there
      try {
        const content = await fs.promises.readFile(filePath, "utf-8");
        if (!content.includes("Real-Time Transcript Ended")) {
          const footer = `\n\n=== Real-Time Transcript Ended at ${new Date().toISOString()} (Duration: ${Math.round(duration / 1000)}s) ===\n`;
          await fs.promises.appendFile(filePath, footer, "utf-8");
        }
      } catch (err) {
        // File might not exist yet, that's okay
      }
      
      console.log(`[ProcessingHelper] Real-time transcription session ended: ${filePath}`);
      
      // Clear session
      this.realTimeTranscriptSession = null;
      
      return filePath;
    } catch (error) {
      console.error("[ProcessingHelper] Error stopping real-time transcription:", error);
      throw error;
    }
  }

  /**
   * Get the current real-time transcript content
   * @returns Full transcript content or null if no session is active
   */
  public async getRealTimeTranscript(): Promise<string | null> {
    if (!this.realTimeTranscriptSession) {
      return null;
    }

    try {
      const content = await fs.promises.readFile(this.realTimeTranscriptSession.filePath, "utf-8");
      return content;
    } catch (error) {
      console.error("[ProcessingHelper] Error reading real-time transcript:", error);
      return null;
    }
  }

  /**
   * Check if a real-time transcription session is active
   * @returns True if session is active
   */
  public isRealTimeTranscriptionActive(): boolean {
    return this.realTimeTranscriptSession !== null;
  }
}
