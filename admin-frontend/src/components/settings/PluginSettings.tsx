'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/Button';
import { RestartModal } from '@/components/settings/RestartModal';

interface PluginConfig {
  [key: string]: string | number | boolean | PluginConfig | string[];
}

interface Plugin {
  name: string;
  friendlyName: string;
  enabled: boolean;
  config: PluginConfig;
  defaults: PluginConfig;
}

interface PluginSettingsProps {
  onRestart: (reason: string) => void;
}

// Helper to convert array to string for display
function arrayToString(value: string[]): string {
  return value.join(', ');
}

// Helper to convert string back to array
function stringToArray(value: string): string[] {
  return value.split(',').map(item => item.trim()).filter(Boolean);
}

// Recursive component to render config fields
function ConfigFields({
  config,
  defaults,
  path,
  pluginName,
  onUpdate,
}: {
  config: PluginConfig;
  defaults: PluginConfig;
  path: string[];
  pluginName: string;
  onUpdate: (key: string[], value: any) => void;
}) {
  return (
    <div className="space-y-4">
      {Object.entries(config).map(([key, value]) => {
        const currentPath = [...path, key];
        const fieldId = `${pluginName}-${currentPath.join('-')}`;
        
        if (typeof value === 'object' && value !== null) {
          if (Array.isArray(value)) {
            return (
              <div key={key} className="space-y-1">
                <label
                  htmlFor={fieldId}
                  className="block text-sm font-medium text-white"
                >
                  {key}
                </label>
                <input
                  type="text"
                  id={fieldId}
                  value={arrayToString(value)}
                  onChange={(e) => onUpdate(currentPath, stringToArray(e.target.value))}
                  className="block w-full rounded-md border border-white/10 bg-white/5 px-3 py-2 text-white placeholder-zinc-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 sm:text-sm"
                />
                {defaults[key] !== undefined && (
                  <p className="text-xs text-zinc-500">
                    Default: {Array.isArray(defaults[key]) ? arrayToString(defaults[key] as string[]) : String(defaults[key])}
                  </p>
                )}
                <p className="text-xs text-zinc-500">
                  Enter values separated by commas
                </p>
              </div>
            );
          }
          
          return (
            <div key={key} className="space-y-2">
              <h4 className="text-sm font-medium text-white capitalize">{key}</h4>
              <div className="pl-4 border-l border-white/10">
                <ConfigFields
                  config={value as PluginConfig}
                  defaults={defaults[key] as PluginConfig || {}}
                  path={currentPath}
                  pluginName={pluginName}
                  onUpdate={onUpdate}
                />
              </div>
            </div>
          );
        }

        return (
          <div key={key} className="space-y-1">
            <label
              htmlFor={fieldId}
              className="block text-sm font-medium text-white"
            >
              {key}
            </label>
            {typeof value === 'boolean' ? (
              <input
                type="checkbox"
                id={fieldId}
                checked={value}
                onChange={(e) => onUpdate(currentPath, e.target.checked)}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-white/10 bg-white/5 rounded"
              />
            ) : (
              <input
                type="text"
                id={fieldId}
                value={value}
                onChange={(e) => onUpdate(currentPath, e.target.value)}
                className="block w-full rounded-md border border-white/10 bg-white/5 px-3 py-2 text-white placeholder-zinc-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 sm:text-sm"
              />
            )}
            {defaults[key] !== undefined && (
              <p className="text-xs text-zinc-500">
                Default: {typeof defaults[key] === 'boolean' ? (defaults[key] ? 'true' : 'false') : String(defaults[key])}
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}

export function PluginSettings({ onRestart }: PluginSettingsProps) {
  const [plugins, setPlugins] = useState<Plugin[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [expandedPlugin, setExpandedPlugin] = useState<string | null>(null);
  const [showRestartModal, setShowRestartModal] = useState(false);
  const [pendingSave, setPendingSave] = useState<Plugin | null>(null);

  useEffect(() => {
    fetch('/api/plugins')
      .then((res) => res.json())
      .then((data) => {
        setPlugins(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load plugins:', err);
        setError('Failed to load plugins');
        setLoading(false);
      });
  }, []);

  const handleSave = async (plugin: Plugin) => {
    setPendingSave(plugin);
    setShowRestartModal(true);
  };

  const handleConfirmSave = async () => {
    if (!pendingSave) return;

    try {
      const res = await fetch(`/api/plugins/${pendingSave.name}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          enabled: pendingSave.enabled,
          config: pendingSave.config,
        }),
      });

      if (!res.ok) throw new Error('Failed to save plugin settings');

      setShowRestartModal(false);
      setPendingSave(null);
    } catch (err) {
      console.error('Failed to save plugin settings:', err);
      setError('Failed to save plugin settings');
    }
  };

  const togglePlugin = (index: number) => {
    setPlugins((prev) => {
      const updated = [...prev];
      updated[index] = {
        ...updated[index],
        enabled: !updated[index].enabled,
      };
      return updated;
    });
  };

  const updatePluginConfig = (
    pluginIndex: number,
    path: string[],
    value: string | boolean | string[]
  ) => {
    setPlugins((prev) => {
      const updated = [...prev];
      let config = { ...updated[pluginIndex].config };
      let current = config;
      let currentExisting = updated[pluginIndex].config;
      
      // Navigate to the nested object
      for (let i = 0; i < path.length - 1; i++) {
        const key = path[i];
        current[key] = typeof current[key] === 'object' ? { ...current[key] } : {};
        current = current[key] as PluginConfig;
        currentExisting = currentExisting[key] as PluginConfig;
      }
      
      const lastKey = path[path.length - 1];
      const existingValue = currentExisting[lastKey];
      
      // Coerce value based on existing type
      if (Array.isArray(existingValue)) {
        current[lastKey] = Array.isArray(value) ? value : stringToArray(value as string);
      } else if (typeof existingValue === 'boolean') {
        current[lastKey] = Boolean(value);
      } else if (typeof existingValue === 'number') {
        current[lastKey] = Number(value);
      } else {
        current[lastKey] = value;
      }

      updated[pluginIndex] = {
        ...updated[pluginIndex],
        config,
      };
      return updated;
    });
  };

  if (loading) {
    return <div className="text-zinc-400">Loading plugins...</div>;
  }

  return (
    <div className="space-y-6">
      <h3 className="text-xl font-semibold text-white">
        Plugin Settings
      </h3>
      <div className="mt-2 max-w-xl text-sm text-zinc-400">
        <p>Configure and enable/disable plugins.</p>
      </div>

      {error && (
        <div className="rounded-md bg-red-500/10 p-4">
          <p className="text-sm text-red-400">{error}</p>
        </div>
      )}

      <div className="mt-6 space-y-4">
        {plugins.map((plugin, index) => (
          <div key={plugin.name} className="glass-card">
            <div
              className="px-4 py-3 flex items-center justify-between cursor-pointer"
              onClick={() =>
                setExpandedPlugin(
                  expandedPlugin === plugin.name ? null : plugin.name
                )
              }
            >
              <div className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  checked={plugin.enabled}
                  onChange={() => togglePlugin(index)}
                  onClick={(e) => e.stopPropagation()}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-white/10 bg-white/5 rounded"
                />
                <span className="font-medium text-white">{plugin.friendlyName}</span>
              </div>
              <button
                type="button"
                className="text-sm text-blue-400 hover:text-blue-300"
              >
                {expandedPlugin === plugin.name ? 'Collapse' : 'Expand'}
              </button>
            </div>

            {expandedPlugin === plugin.name && (
              <div className="px-4 py-3 border-t border-white/10">
                <div className="space-y-4">
                  <ConfigFields
                    config={plugin.config}
                    defaults={plugin.defaults}
                    path={[]}
                    pluginName={plugin.name}
                    onUpdate={(path, value) => updatePluginConfig(index, path, value)}
                  />
                  <div className="pt-3">
                    <Button
                      type="button"
                      onClick={() => handleSave(plugin)}
                      size="sm"
                    >
                      Save Changes
                    </Button>
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
      <RestartModal
        isOpen={showRestartModal}
        onClose={() => {
          setShowRestartModal(false);
          setPendingSave(null);
        }}
        onConfirm={handleConfirmSave}
        reason="Saving these settings will restart the server. Any active processing on the server will be cancelled"
      />
    </div>
  );
} 