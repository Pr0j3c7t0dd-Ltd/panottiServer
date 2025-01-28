import { NextResponse } from 'next/server';
import https from 'https';

export async function GET() {
  try {
    const agent = new https.Agent({
      rejectUnauthorized: false
    });

    const response = await fetch(`${process.env.NEXT_PUBLIC_API_BASE_URL}/health`, {
      headers: {
        'X-API-Key': process.env.NEXT_PUBLIC_API_KEY || '',
        'Accept': 'application/json',
      },
      // @ts-ignore
      agent: agent
    });

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Health check error:', error);
    return NextResponse.json(
      { error: 'Failed to check server health' },
      { status: 500 }
    );
  }
} 