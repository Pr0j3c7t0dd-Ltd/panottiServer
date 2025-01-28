import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';
import yaml from 'js-yaml';

const PLUGINS_DIR = path.join(process.cwd(), '..', 'app', 'plugins');

export async function GET() {
  try {
    // Read all plugin directories
    const pluginDirs = await fs.readdir(PLUGINS_DIR);
    
    // Process each plugin
    const plugins = await Promise.all(
      pluginDirs.map(async (dir) => {
        const pluginPath = path.join(PLUGINS_DIR, dir);
        const stat = await fs.stat(pluginPath);
        
        if (!stat.isDirectory()) return null;

        try {
          // Read both plugin.yaml and plugin.yaml.example
          const [configContent, defaultContent] = await Promise.all([
            fs.readFile(path.join(pluginPath, 'plugin.yaml'), 'utf-8'),
            fs.readFile(path.join(pluginPath, 'plugin.yaml.example'), 'utf-8'),
          ]);

          const config = yaml.load(configContent) as any;
          const defaults = yaml.load(defaultContent) as any;

          return {
            name: dir,
            friendlyName: dir.split('_').map(word => 
              word.charAt(0).toUpperCase() + word.slice(1)
            ).join(' '),
            enabled: config.enabled ?? false,
            config: config.config ?? {},
            defaults: defaults.config ?? {},
          };
        } catch (err) {
          console.error(`Error reading plugin ${dir}:`, err);
          return null;
        }
      })
    );

    // Filter out null values and return valid plugins
    return NextResponse.json(plugins.filter(Boolean));
  } catch (error) {
    console.error('Failed to read plugins:', error);
    return NextResponse.json(
      { success: false, message: 'Failed to read plugins' },
      { status: 500 }
    );
  }
} 