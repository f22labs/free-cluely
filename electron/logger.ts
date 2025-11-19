const formatPrefix = (level: string) => {
  const timestamp = new Date().toISOString();
  return `[${timestamp}] [${level}]`;
};

type LogMethod = (...args: unknown[]) => void;

const createLoggerMethod =
  (level: "INFO" | "WARN" | "ERROR" | "DEBUG", consoleMethod: (...args: any[]) => void): LogMethod =>
  (...args: unknown[]) => {
    consoleMethod(formatPrefix(level), ...args);
  };

export const logger = {
  info: createLoggerMethod("INFO", console.log),
  warn: createLoggerMethod("WARN", console.warn),
  error: createLoggerMethod("ERROR", console.error),
  debug: createLoggerMethod("DEBUG", console.debug ?? console.log),
};

export type Logger = typeof logger;

export default logger;

