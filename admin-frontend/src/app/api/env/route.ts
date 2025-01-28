import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';
import dotenv from 'dotenv';

// Helper function to read .env file
async function readEnvFile(filePath: string) {
  try {
    const content = await fs.readFile(filePath, 'utf-8');
    return dotenv.parse(content);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return {};
    }
    throw error;
  }
}

export async function GET() {
  try {
    const envPath = path.join(process.cwd(), '..', '.env');
    const envExamplePath = path.join(process.cwd(), '..', '.env.example');

    const [env, defaults] = await Promise.all([
      readEnvFile(envPath),
      readEnvFile(envExamplePath)
    ]);

    return NextResponse.json({ env, defaults });
  } catch (error) {
    console.error('Failed to read environment files:', error);
    return NextResponse.json(
      { success: false, message: 'Failed to read environment files' },
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  try {
    const { env } = await request.json();
    const envPath = path.join(process.cwd(), '..', '.env');

    // Convert env object to string format
    const envContent = Object.entries(env)
      .map(([key, value]) => `${key}=${value}`)
      .join('\n');

    // Write to .env file
    await fs.writeFile(envPath, envContent, 'utf-8');

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Failed to update environment variables:', error);
    return NextResponse.json(
      { success: false, message: 'Failed to update environment variables' },
      { status: 500 }
    );
  }
}
