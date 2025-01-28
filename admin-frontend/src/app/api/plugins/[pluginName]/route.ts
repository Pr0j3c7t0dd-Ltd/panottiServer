import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';
import yaml from 'js-yaml';

const PLUGINS_DIR = path.join(process.cwd(), '..', 'app', 'plugins');

// Helper function to coerce types based on existing config
function coerceConfigValues(newConfig: any, existingConfig: any): any {
  if (!existingConfig || typeof existingConfig !== 'object') {
    return newConfig;
  }

  const result: any = {};
  for (const [key, value] of Object.entries(newConfig)) {
    const existingValue = existingConfig[key];
    
    if (Array.isArray(existingValue)) {
      // Ensure array values are properly handled
      result[key] = Array.isArray(value) ? value : 
        (typeof value === 'string' ? value.split(',').map(item => item.trim()).filter(Boolean) : [String(value)]);
    } else if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      // Recursively handle nested objects
      result[key] = coerceConfigValues(value, existingValue || {});
    } else if (typeof existingValue === 'boolean') {
      result[key] = Boolean(value);
    } else if (typeof existingValue === 'number') {
      result[key] = Number(value);
    } else {
      result[key] = value;
    }
  }
  return result;
}

export async function POST(
  request: Request,
  { params }: { params: { pluginName: string } }
) {
  try {
    const { enabled, config } = await request.json();
    const pluginPath = path.join(PLUGINS_DIR, params.pluginName);

    // Read existing plugin.yaml
    const configPath = path.join(pluginPath, 'plugin.yaml');
    const existingContent = await fs.readFile(configPath, 'utf-8');
    const existingConfig = yaml.load(existingContent) as any;

    // Update configuration with proper type coercion
    const updatedConfig = {
      ...existingConfig,
      enabled: Boolean(enabled),
      config: coerceConfigValues(config, existingConfig.config || {}),
    };

    // Write updated configuration with proper YAML formatting
    await fs.writeFile(
      configPath,
      yaml.dump(updatedConfig, { 
        indent: 2, 
        lineWidth: -1, 
        noRefs: true,
        quotingType: '"',  // Use double quotes for strings
        forceQuotes: true,  // Force quotes around strings
        styles: {
          '!!seq': 'flow'  // Use flow style for sequences (arrays)
        }
      })
    );

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Failed to update plugin:', error);
    return NextResponse.json(
      { success: false, message: 'Failed to update plugin' },
      { status: 500 }
    );
  }
} 