'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/Button';

interface EnvVars {
  [key: string]: string;
}

interface EnvironmentVariablesProps {
  onRestart: (reason: string) => void;
}

export function EnvironmentVariables({ onRestart }: EnvironmentVariablesProps) {
  const [envVars, setEnvVars] = useState<EnvVars>({});
  const [defaultVars, setDefaultVars] = useState<EnvVars>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch('/api/env')
      .then((res) => res.json())
      .then((data) => {
        setEnvVars(data.env);
        setDefaultVars(data.defaults);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load environment variables:', err);
        setError('Failed to load environment variables');
        setLoading(false);
      });
  }, []);

  const handleSave = async () => {
    try {
      const res = await fetch('/api/env', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ env: envVars }),
      });

      if (!res.ok) throw new Error('Failed to save environment variables');

      onRestart('Environment variables have been updated');
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
            <input
              type="text"
              id={key}
              value={value}
              onChange={(e) =>
                setEnvVars((prev) => ({ ...prev, [key]: e.target.value }))
              }
              className="block w-full rounded-md border border-white/10 bg-white/5 px-3 py-2 text-white placeholder-zinc-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 sm:text-sm"
            />
            {defaultVars[key] && (
              <p className="text-xs text-zinc-500">
                Default: {defaultVars[key]}
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
    </div>
  );
} 