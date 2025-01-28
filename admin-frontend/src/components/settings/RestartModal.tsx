'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/Button';

interface RestartModalProps {
  isOpen: boolean;
  onClose: () => void;
  reason: string;
}

export function RestartModal({ isOpen, onClose, reason }: RestartModalProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  if (!isOpen) return null;

  const handleRestart = async () => {
    setLoading(true);
    setError('');

    try {
      const res = await fetch('/api/restart', {
        method: 'POST',
      });

      if (!res.ok) throw new Error('Failed to restart server');

      onClose();
    } catch (err) {
      console.error('Failed to restart server:', err);
      setError('Failed to restart server');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm overflow-y-auto h-full w-full flex items-center justify-center">
      <div className="relative glass-card p-8 m-4 max-w-xl w-full">
        <h2 className="text-xl font-bold text-white mb-4">Restart Required</h2>
        <p className="text-zinc-400 mb-6">
          {reason}. The server needs to be restarted for these changes to take effect.
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
            {loading ? 'Restarting...' : 'Restart Server'}
          </Button>
        </div>
      </div>
    </div>
  );
} 