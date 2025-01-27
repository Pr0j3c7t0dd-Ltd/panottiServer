import { useState, useCallback } from 'react';
import logger from '@/lib/logger';

interface ErrorBoundaryState {
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
}

export const useErrorBoundary = (fallback?: React.ReactNode) => {
  const [errorState, setErrorState] = useState<ErrorBoundaryState>({
    error: null,
    errorInfo: null,
  });

  const handleError = useCallback((error: Error, errorInfo: React.ErrorInfo) => {
    setErrorState({ error, errorInfo });
    logger.error('Error caught by ErrorBoundary:', {
      error: error.message,
      stack: errorInfo.componentStack,
    });
  }, []);

  const resetError = useCallback(() => {
    setErrorState({ error: null, errorInfo: null });
  }, []);

  return {
    error: errorState.error,
    errorInfo: errorState.errorInfo,
    handleError,
    resetError,
  };
};

export default useErrorBoundary;
