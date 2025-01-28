'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/Button';

interface RestartModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  reason: string;
}

export function RestartModal({ isOpen, onClose, onConfirm, reason }: RestartModalProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  if (!isOpen) return null;

  const handleRestart = async () => {
    setLoading(true);
    setError('');

    try {
      await onConfirm();
      onClose();
    } catch (err) {
      console.error('Failed to save changes:', err);
      setError('Failed to save changes');
    } finally {
      setLoading(false);
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
          >
            Cancel
          </Button>
          <Button
            onClick={handleRestart}
            disabled={loading}
          >
            {loading ? 'Saving...' : 'Save Changes'}
          </Button>
        </div>
      </div>
    </div>
  );
} 