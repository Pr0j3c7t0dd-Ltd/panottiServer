'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/Button';
import { RestartModal } from '@/components/settings/RestartModal';

interface EnvVars {
  [key: string]: string | boolean;
}

interface EnvironmentVariablesProps {
  onRestart: (reason: string) => void;
}

// Helper to detect if a value is boolean
function isBooleanString(value: string): boolean {
  return value.toLowerCase() === 'true' || value.toLowerCase() === 'false';
}

// Helper to convert string to boolean
function stringToBoolean(value: string): boolean {
  return value.toLowerCase() === 'true';
}

export function EnvironmentVariables({ onRestart }: EnvironmentVariablesProps) {
  const [envVars, setEnvVars] = useState<EnvVars>({});
  const [defaultVars, setDefaultVars] = useState<EnvVars>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [showModal, setShowModal] = useState(false);

  useEffect(() => {
    fetch('/api/env')
      .then((res) => res.json())
      .then((data) => {
        // Convert string boolean values to actual booleans
        const processedEnv = Object.entries(data.env).reduce((acc, [key, value]) => {
          acc[key] = isBooleanString(String(value)) ? stringToBoolean(String(value)) : String(value);
          return acc;
        }, {} as EnvVars);

        const processedDefaults = Object.entries(data.defaults).reduce((acc, [key, value]) => {
          acc[key] = isBooleanString(String(value)) ? stringToBoolean(String(value)) : String(value);
          return acc;
        }, {} as EnvVars);

        setEnvVars(processedEnv);
        setDefaultVars(processedDefaults);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load environment variables:', err);
        setError('Failed to load environment variables');
        setLoading(false);
      });
  }, []);

  const handleSave = async () => {
    setShowModal(true);
  };

  const handleConfirmSave = async () => {
    try {
      const res = await fetch('/api/env', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ env: envVars }),
      });

      if (!res.ok) throw new Error('Failed to save environment variables');

      setShowModal(false);
    } catch (err) {
      console.error('Failed to save environment variables:', err);
      setError('Failed to save environment variables');
    }
  };

  if (loading) {
    return <div className="text-zinc-400">Loading environment variables...</div>;
  }

  return (
    <div>
      <h3 className="text-xl font-semibold text-white">
        Environment Variables
      </h3>
      <div className="mt-2 max-w-xl text-sm text-zinc-400">
        <p>Configure your application's environment variables.</p>
      </div>

      {error && (
        <div className="mt-2 text-sm text-red-500">{error}</div>
      )}

      <div className="mt-6 space-y-4">
        {Object.entries(envVars).map(([key, value]) => (
          <div key={key} className="space-y-1">
            <label
              htmlFor={key}
              className="block text-sm font-medium text-white"
            >
              {key}
            </label>
            {typeof value === 'boolean' ? (
              <input
                type="checkbox"
                id={key}
                checked={value}
                onChange={(e) =>
                  setEnvVars((prev) => ({ ...prev, [key]: e.target.checked }))
                }
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-white/10 bg-white/5 rounded"
              />
            ) : (
              <input
                type="text"
                id={key}
                value={value}
                onChange={(e) =>
                  setEnvVars((prev) => ({ ...prev, [key]: e.target.value }))
                }
                className="block w-full rounded-md border border-white/10 bg-white/5 px-3 py-2 text-white placeholder-zinc-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 sm:text-sm"
              />
            )}
            {defaultVars[key] !== undefined && (
              <p className="text-xs text-zinc-500">
                Default: {typeof defaultVars[key] === 'boolean' ? (defaultVars[key] ? 'true' : 'false') : String(defaultVars[key])}
              </p>
            )}
          </div>
        ))}
      </div>

      <div className="mt-6">
        <Button
          type="button"
          onClick={handleSave}
        >
          Save Changes
        </Button>
      </div>

      <RestartModal
        isOpen={showModal}
        onClose={() => setShowModal(false)}
        onConfirm={handleConfirmSave}
        reason="For these changes to take effect, you must stop the server and start it again in a new shell, as environment variables cannot be updated in a running process"
      />
    </div>
  );
} 