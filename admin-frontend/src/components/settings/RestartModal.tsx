'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/Button';

interface RestartModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  reason: string;
  isLoading?: boolean;
}

export function RestartModal({ isOpen, onClose, onConfirm, reason, isLoading = false }: RestartModalProps) {
  const [error, setError] = useState('');

  if (!isOpen) return null;

  const handleRestart = async () => {
    setError('');

    try {
      await onConfirm();
    } catch (err) {
      console.error('Failed to save changes:', err);
      setError('Failed to save changes');
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm overflow-y-auto h-full w-full flex items-center justify-center z-[100]">
      <div className="relative glass-card p-8 m-4 max-w-xl w-full">
        <h2 className="text-xl font-bold text-white mb-4">Save Changes</h2>
        <p className="text-zinc-400 mb-6">
          {reason}
        </p>

        {error && (
          <div className="mb-4 text-sm text-red-500">{error}</div>
        )}

        <div className="flex justify-end space-x-3">
          <Button
            variant="secondary"
            onClick={onClose}
            disabled={isLoading}
          >
            Cancel
          </Button>
          <Button
            onClick={handleRestart}
            disabled={isLoading}
          >
            {isLoading ? (
              <div className="flex items-center space-x-2">
                <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Saving...</span>
              </div>
            ) : 'Save Changes'}
          </Button>
        </div>
      </div>
    </div>
  );
} 