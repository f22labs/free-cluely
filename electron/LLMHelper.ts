import { GoogleGenerativeAI, GenerativeModel } from "@google/generative-ai"
import fs from "fs"
import { logger } from "./logger"

const metricsLoggingEnabled = process.env.RTSTT_METRICS_LOG === "1"

interface OllamaResponse {
  response: string
  done: boolean
}

export class LLMHelper {
  private model: GenerativeModel | null = null
  private readonly systemPrompt = `You are Wingman AI, a helpful, proactive assistant for any kind of problem or situation (not just coding). For any user input, analyze the situation, provide a clear problem statement, relevant context, and suggest several possible responses or actions the user could take next. Always explain your reasoning. Present your suggestions as a list of options or next steps.`
  private useOllama: boolean = false
  private ollamaModel: string = "llama3.2"
  private ollamaUrl: string = "http://localhost:11434"

  constructor(apiKey?: string, useOllama: boolean = false, ollamaModel?: string, ollamaUrl?: string) {
    this.useOllama = useOllama
    
    if (useOllama) {
      this.ollamaUrl = ollamaUrl || "http://localhost:11434"
      this.ollamaModel = ollamaModel || "gemma:latest" // Default fallback
      logger.info("[LLMHelper] Using Ollama with model:", this.ollamaModel)
      
      // Auto-detect and use first available model if specified model doesn't exist
      this.initializeOllamaModel()
    } else if (apiKey) {
      const genAI = new GoogleGenerativeAI(apiKey)
      this.model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" })
      logger.info("[LLMHelper] Using Google Gemini")
    } else {
      throw new Error("Either provide Gemini API key or enable Ollama mode")
    }
  }

  private async fileToGenerativePart(imagePath: string) {
    const imageData = await fs.promises.readFile(imagePath)
    return {
      inlineData: {
        data: imageData.toString("base64"),
        mimeType: "image/png"
      }
    }
  }

  private cleanJsonResponse(text: string): string {
    // Remove markdown code block syntax if present
    text = text.replace(/^```(?:json)?\n/, '').replace(/\n```$/, '');
    // Remove any leading/trailing whitespace
    text = text.trim();
    return text;
  }

  private async callOllama(prompt: string): Promise<string> {
    try {
      const response = await fetch(`${this.ollamaUrl}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: this.ollamaModel,
          prompt: prompt,
          stream: false,
          options: {
            temperature: 0.7,
            top_p: 0.9,
          }
        }),
      })

      if (!response.ok) {
        throw new Error(`Ollama API error: ${response.status} ${response.statusText}`)
      }

      const data: OllamaResponse = await response.json()
      return data.response
    } catch (error) {
      logger.error("[LLMHelper] Error calling Ollama:", error)
      throw new Error(`Failed to connect to Ollama: ${error.message}. Make sure Ollama is running on ${this.ollamaUrl}`)
    }
  }

  private async checkOllamaAvailable(): Promise<boolean> {
    try {
      const response = await fetch(`${this.ollamaUrl}/api/tags`)
      return response.ok
    } catch {
      return false
    }
  }

  private async initializeOllamaModel(): Promise<void> {
    try {
      const availableModels = await this.getOllamaModels()
      if (availableModels.length === 0) {
        logger.warn("[LLMHelper] No Ollama models found")
        return
      }

      // Check if current model exists, if not use the first available
      if (!availableModels.includes(this.ollamaModel)) {
        this.ollamaModel = availableModels[0]
        logger.info("[LLMHelper] Auto-selected first available model:", this.ollamaModel)
      }

      // Test the selected model works
      const testResult = await this.callOllama("Hello")
      logger.info("[LLMHelper] Successfully initialized with model:", this.ollamaModel)
    } catch (error) {
      logger.error("[LLMHelper] Failed to initialize Ollama model:", error.message)
      // Try to use first available model as fallback
      try {
        const models = await this.getOllamaModels()
        if (models.length > 0) {
          this.ollamaModel = models[0]
          logger.info("[LLMHelper] Fallback to:", this.ollamaModel)
        }
      } catch (fallbackError) {
        logger.error("[LLMHelper] Fallback also failed:", fallbackError.message)
      }
    }
  }

  public async extractProblemFromImages(imagePaths: string[]) {
    try {
      const imageParts = await Promise.all(imagePaths.map(path => this.fileToGenerativePart(path)))
      
      const prompt = `${this.systemPrompt}\n\nYou are a wingman. Please analyze these images and extract the following information in JSON format:\n{
  "problem_statement": "A clear statement of the problem or situation depicted in the images.",
  "context": "Relevant background or context from the images.",
  "suggested_responses": ["First possible answer or action", "Second possible answer or action", "..."],
  "reasoning": "Explanation of why these suggestions are appropriate."
}\nImportant: Return ONLY the JSON object, without any markdown formatting or code blocks.`

      const result = await this.model.generateContent([prompt, ...imageParts])
      const response = await result.response
      const text = this.cleanJsonResponse(response.text())
      return JSON.parse(text)
    } catch (error) {
      logger.error("Error extracting problem from images:", error)
      throw error
    }
  }

  public async generateSolution(problemInfo: any) {
    const prompt = `${this.systemPrompt}\n\nGiven this problem or situation:\n${JSON.stringify(problemInfo, null, 2)}\n\nPlease provide your response in the following JSON format:\n{
  "solution": {
    "code": "The code or main answer here.",
    "problem_statement": "Restate the problem or situation.",
    "context": "Relevant background/context.",
    "suggested_responses": ["First possible answer or action", "Second possible answer or action", "..."],
    "reasoning": "Explanation of why these suggestions are appropriate."
  }
}\nImportant: Return ONLY the JSON object, without any markdown formatting or code blocks.`

    logger.info("[LLMHelper] Calling Gemini LLM for solution...");
    try {
      const result = await this.model.generateContent(prompt)
      logger.info("[LLMHelper] Gemini LLM returned result.");
      const response = await result.response
      const text = this.cleanJsonResponse(response.text())
      const parsed = JSON.parse(text)
      logger.debug("[LLMHelper] Parsed LLM response:", parsed)
      return parsed
    } catch (error) {
      logger.error("[LLMHelper] Error in generateSolution:", error);
      throw error;
    }
  }

  public async debugSolutionWithImages(problemInfo: any, currentCode: string, debugImagePaths: string[]) {
    try {
      const imageParts = await Promise.all(debugImagePaths.map(path => this.fileToGenerativePart(path)))
      
      const prompt = `${this.systemPrompt}\n\nYou are a wingman. Given:\n1. The original problem or situation: ${JSON.stringify(problemInfo, null, 2)}\n2. The current response or approach: ${currentCode}\n3. The debug information in the provided images\n\nPlease analyze the debug information and provide feedback in this JSON format:\n{
  "solution": {
    "code": "The code or main answer here.",
    "problem_statement": "Restate the problem or situation.",
    "context": "Relevant background/context.",
    "suggested_responses": ["First possible answer or action", "Second possible answer or action", "..."],
    "reasoning": "Explanation of why these suggestions are appropriate."
  }
}\nImportant: Return ONLY the JSON object, without any markdown formatting or code blocks.`

      const result = await this.model.generateContent([prompt, ...imageParts])
      const response = await result.response
      const text = this.cleanJsonResponse(response.text())
      const parsed = JSON.parse(text)
      logger.debug("[LLMHelper] Parsed debug LLM response:", parsed)
      return parsed
    } catch (error) {
      logger.error("Error debugging solution with images:", error)
      throw error
    }
  }

  public async analyzeAudioFile(audioPath: string) {
    try {
      const audioData = await fs.promises.readFile(audioPath);
      const audioPart = {
        inlineData: {
          data: audioData.toString("base64"),
          mimeType: "audio/mp3"
        }
      };
      const prompt = `${this.systemPrompt}\n\nDescribe this audio clip in a short, concise answer. In addition to your main answer, suggest several possible actions or responses the user could take next based on the audio. Do not return a structured JSON object, just answer naturally as you would to a user.`;
      const result = await this.model.generateContent([prompt, audioPart]);
      const response = await result.response;
      const text = response.text();
      return { text, timestamp: Date.now() };
    } catch (error) {
      logger.error("Error analyzing audio file:", error);
      throw error;
    }
  }

  public async analyzeAudioFromBase64(data: string, mimeType: string) {
    try {
      const audioPart = {
        inlineData: {
          data,
          mimeType
        }
      };
      const prompt = `${this.systemPrompt}\n\nDescribe this audio clip in a short, concise answer. In addition to your main answer, suggest several possible actions or responses the user could take next based on the audio. Do not return a structured JSON object, just answer naturally as you would to a user and be concise.`;
      const result = await this.model.generateContent([prompt, audioPart]);
      const response = await result.response;
      const text = response.text();
      return { text, timestamp: Date.now() };
    } catch (error) {
      logger.error("Error analyzing audio from base64:", error);
      throw error;
    }
  }

  public async analyzeImageFile(imagePath: string) {
    try {
      const imageData = await fs.promises.readFile(imagePath);
      const imagePart = {
        inlineData: {
          data: imageData.toString("base64"),
          mimeType: "image/png"
        }
      };
      const prompt = `${this.systemPrompt}\n\nDescribe the content of this image in a short, concise answer. In addition to your main answer, suggest several possible actions or responses the user could take next based on the image. Do not return a structured JSON object, just answer naturally as you would to a user. Be concise and brief.`;
      const result = await this.model.generateContent([prompt, imagePart]);
      const response = await result.response;
      const text = response.text();
      return { text, timestamp: Date.now() };
    } catch (error) {
      logger.error("Error analyzing image file:", error);
      throw error;
    }
  }

  /**
   * Transcribe audio file with a custom prompt
   * @param audioPath Path to the audio file
   * @param customPrompt Optional custom prompt for transcription. If not provided, uses default transcription prompt.
   * @returns Transcript text and timestamp
   */
  public async transcribeAudioFile(audioPath: string, customPrompt?: string): Promise<{ text: string; timestamp: number }> {
    try {
      const audioData = await fs.promises.readFile(audioPath);
      const audioPart = {
        inlineData: {
          data: audioData.toString("base64"),
          mimeType: audioPath.endsWith('.wav') ? "audio/wav" : "audio/mp3"
        }
      };
      
      // Use custom prompt or default transcription prompt
      const prompt = customPrompt || `Please transcribe this audio file word-for-word. Provide a clear, accurate transcription of everything that is said. If there are multiple speakers, indicate who is speaking. Return only the transcription text, no additional commentary.`;
      
      const result = await this.model.generateContent([prompt, audioPart]);
      const response = await result.response;
      const text = response.text();
      return { text, timestamp: Date.now() };
    } catch (error) {
      logger.error("Error transcribing audio file:", error);
      throw error;
    }
  }

  /**
   * Transcribe audio from base64 data with a custom prompt
   * @param data Base64 encoded audio data
   * @param mimeType MIME type of the audio
   * @param customPrompt Optional custom prompt for transcription
   * @returns Transcript text and timestamp
   */
  public async transcribeAudioFromBase64(data: string, mimeType: string, customPrompt?: string): Promise<{ text: string; timestamp: number }> {
    try {
      const audioPart = {
        inlineData: {
          data,
          mimeType
        }
      };
      
      // Use custom prompt or default transcription prompt
      const prompt = customPrompt || `Please transcribe this audio clip word-for-word. Provide a clear, accurate transcription of everything that is said. If there are multiple speakers, indicate who is speaking. Return only the transcription text, no additional commentary.`;
      
      const result = await this.model.generateContent([prompt, audioPart]);
      const response = await result.response;
      const text = response.text();
      return { text, timestamp: Date.now() };
    } catch (error) {
      logger.error("Error transcribing audio from base64:", error);
      throw error;
    }
  }

  public async chatWithGemini(message: string): Promise<string> {
    try {
      if (this.useOllama) {
        return this.callOllama(message);
      } else if (this.model) {
        const result = await this.model.generateContent(message);
        const response = await result.response;
        return response.text();
      } else {
        throw new Error("No LLM provider configured");
      }
    } catch (error) {
      logger.error("[LLMHelper] Error in chatWithGemini:", error);
      throw error;
    }
  }

  public async chat(message: string): Promise<string> {
    return this.chatWithGemini(message);
  }

  public isUsingOllama(): boolean {
    return this.useOllama;
  }

  public async getOllamaModels(): Promise<string[]> {
    if (!this.useOllama) return [];
    
    try {
      const response = await fetch(`${this.ollamaUrl}/api/tags`);
      if (!response.ok) throw new Error('Failed to fetch models');
      
      const data = await response.json();
      return data.models?.map((model: any) => model.name) || [];
    } catch (error) {
      logger.error("[LLMHelper] Error fetching Ollama models:", error);
      return [];
    }
  }

  public getCurrentProvider(): "ollama" | "gemini" {
    return this.useOllama ? "ollama" : "gemini";
  }

  public getCurrentModel(): string {
    return this.useOllama ? this.ollamaModel : "gemini-2.0-flash";
  }

  public async switchToOllama(model?: string, url?: string): Promise<void> {
    this.useOllama = true;
    if (url) this.ollamaUrl = url;
    
    if (model) {
      this.ollamaModel = model;
    } else {
      // Auto-detect first available model
      await this.initializeOllamaModel();
    }
    
    logger.info("[LLMHelper] Switched to Ollama:", this.ollamaModel, "at", this.ollamaUrl);
  }

  public async switchToGemini(apiKey?: string): Promise<void> {
    if (apiKey) {
      const genAI = new GoogleGenerativeAI(apiKey);
      this.model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });
    }
    
    if (!this.model && !apiKey) {
      throw new Error("No Gemini API key provided and no existing model instance");
    }
    
    this.useOllama = false;
    logger.info("[LLMHelper] Switched to Gemini");
  }

  public async testConnection(): Promise<{ success: boolean; error?: string }> {
    try {
      if (this.useOllama) {
        const available = await this.checkOllamaAvailable();
        if (!available) {
          return { success: false, error: `Ollama not available at ${this.ollamaUrl}` };
        }
        // Test with a simple prompt
        await this.callOllama("Hello");
        return { success: true };
      } else {
        if (!this.model) {
          return { success: false, error: "No Gemini model configured" };
        }
        // Test with a simple prompt
        const result = await this.model.generateContent("Hello");
        const response = await result.response;
        const text = response.text(); // Ensure the response is valid
        if (text) {
          return { success: true };
        } else {
          return { success: false, error: "Empty response from Gemini" };
        }
      }
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  /**
   * Retry a function with exponential backoff for rate limit errors
   * @param fn Function to retry
   * @param maxRetries Maximum number of retries
   * @param baseDelay Base delay in milliseconds
   * @returns Result of the function
   */
  private async retryWithBackoff<T>(
    fn: () => Promise<T>,
    maxRetries: number = 3,
    baseDelay: number = 1000
  ): Promise<T> {
    let lastError: any;
    
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await fn();
      } catch (error: any) {
        lastError = error;
        
        // Check error status code and message
        const statusCode = error.status || error.statusCode || error.code;
        const errorMessage = error.message || String(error) || "";
        const errorString = errorMessage.toLowerCase();
        
        // Check if it's a rate limit error (429)
        const isRateLimit = statusCode === 429 || 
                           errorString.includes("429") || 
                           errorString.includes("too many requests") ||
                           errorString.includes("resource exhausted") ||
                           errorString.includes("rate limit");
        
        // Check if it's a quota error
        const isQuotaError = errorString.includes("quota") || 
                            errorString.includes("quota exceeded");
        
        // Check if it's a temporary server error (5xx)
        const isServerError = statusCode >= 500 && statusCode < 600;
        
        // Retry on rate limit, quota, or server errors
        if ((isRateLimit || isQuotaError || isServerError) && attempt < maxRetries) {
          const delay = baseDelay * Math.pow(2, attempt); // Exponential backoff: 1s, 2s, 4s
          const errorType = isRateLimit ? "rate limit" : isQuotaError ? "quota" : "server error";
          logger.warn(`[LLMHelper] ${errorType} error (attempt ${attempt + 1}/${maxRetries + 1}), retrying in ${delay}ms...`);
          await new Promise(resolve => setTimeout(resolve, delay));
          continue;
        }
        
        // For other errors or if we've exhausted retries, throw immediately
        throw error;
      }
    }
    
    throw lastError;
  }

  /**
   * Generate meeting suggestions based on transcript and system prompt
   * @param transcript Current meeting transcript
   * @param systemPrompt Context about the person and meeting
   * @returns Suggestion text and type
   */
  public async generateMeetingSuggestion(
    transcript: string,
    systemPrompt: string
  ): Promise<{
    text: string;
    type: "response" | "question" | "negotiation";
    metrics?: {
      provider: "ollama" | "gemini";
      model: string;
      attempts: number;
      llm_duration_ms: number;
      llm_started_at: string;
      llm_completed_at: string;
    };
  }> {
    try {
      if (metricsLoggingEnabled) {
        logger.info("[LLMHelper][Metrics] generateMeetingSuggestion invoked");
      }
      const llmCallStartEpochMs = Date.now();
      const llmCallStartIso = new Date(llmCallStartEpochMs).toISOString();
      let attempts = 0;

      const prompt = `You are an AI meeting assistant helping during a live meeting. 

SYSTEM CONTEXT:
${systemPrompt}

CURRENT MEETING TRANSCRIPT:
${transcript}

TASK:
Based on the conversation so far, provide a helpful suggestion for what the person should say or do next. Consider:
1. If the client asked a question, suggest an empathetic and technically accurate response
2. If there's a negotiation point, suggest a balanced approach
3. If the conversation needs direction, suggest a question or topic to bring up
4. Be concise (2-3 sentences max)
5. Be empathetic to client concerns while maintaining technical accuracy
6. Consider the context of the meeting and the person's role

Provide ONLY the suggestion text, no labels or prefixes. Be natural and conversational.`;

      // Use retry logic with exponential backoff for rate limit errors
      if (metricsLoggingEnabled) {
        logger.info("[LLMHelper][Metrics] Calling LLM with retry (max 3 retries)");
      }
      const responseText = await this.retryWithBackoff(async () => {
        attempts += 1;
        if (this.useOllama) {
          return await this.callOllama(prompt);
        } else if (this.model) {
          const result = await this.model.generateContent(prompt);
          const response = await result.response;
          return response.text();
        } else {
          throw new Error("No LLM provider configured");
        }
      }, 3, 1000); // 3 retries, starting with 1 second delay

      // Determine suggestion type based on content
      const lowerText = responseText.toLowerCase();
      let type: "response" | "question" | "negotiation" = "response";
      
      if (lowerText.includes("?") || lowerText.includes("ask") || lowerText.includes("question")) {
        type = "question";
      } else if (
        lowerText.includes("price") || 
        lowerText.includes("cost") || 
        lowerText.includes("deal") || 
        lowerText.includes("negotiate") ||
        lowerText.includes("discount") ||
        lowerText.includes("contract")
      ) {
        type = "negotiation";
      }

      const llmCallCompletedEpochMs = Date.now();
      const llmCallCompletedIso = new Date(llmCallCompletedEpochMs).toISOString();
      const durationMs = llmCallCompletedEpochMs - llmCallStartEpochMs;

      if (metricsLoggingEnabled) {
        logger.info(
          `[LLMHelper][Metrics] generateMeetingSuggestion finished in ${durationMs}ms (attempts=${attempts}, provider=${this.useOllama ? "ollama" : "gemini"})`
        );
      }

      return {
        text: responseText.trim(),
        type,
        metrics: {
          provider: this.useOllama ? "ollama" : "gemini",
          model: this.useOllama ? this.ollamaModel : "gemini-2.0-flash",
          attempts,
          llm_duration_ms: durationMs,
          llm_started_at: llmCallStartIso,
          llm_completed_at: llmCallCompletedIso
        }
      };
    } catch (error: any) {
      logger.error("[LLMHelper] Error generating meeting suggestion:", error);
      
      // Provide user-friendly error messages
      if (error.message?.includes("429") || error.message?.includes("Too Many Requests")) {
        throw new Error("API rate limit exceeded. Please wait a moment and try again. The system will automatically retry.");
      } else if (error.message?.includes("quota") || error.message?.includes("Quota exceeded")) {
        throw new Error("API quota exceeded. Please check your API usage limits.");
      } else {
        throw error;
      }
    }
  }
} 