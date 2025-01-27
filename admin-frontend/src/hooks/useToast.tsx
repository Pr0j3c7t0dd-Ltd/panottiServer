import { useState, useCallback } from 'react';
import { Toast } from '@/components/ui/Toast';

interface ToastOptions {
  title: string;
  message: string;
  type: 'success' | 'error';
  duration?: number;
}

export function useToast() {
  const [isVisible, setIsVisible] = useState(false);
  const [toastProps, setToastProps] = useState<ToastOptions>({
    title: '',
    message: '',
    type: 'success',
  });

  const hideToast = useCallback(() => {
    setIsVisible(false);
  }, []);

  const showToast = useCallback(
    ({ title, message, type, duration = 5000 }: ToastOptions) => {
      setToastProps({ title, message, type });
      setIsVisible(true);

      const timer = setTimeout(() => {
        hideToast();
      }, duration);

      return () => clearTimeout(timer);
    },
    [hideToast]
  );

  const ToastComponent = (
    <Toast
      show={isVisible}
      title={toastProps.title}
      message={toastProps.message}
      type={toastProps.type}
      onClose={hideToast}
    />
  );

  return { showToast, ToastComponent };
}
