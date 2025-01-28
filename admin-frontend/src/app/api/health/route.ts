import { NextResponse } from 'next/server';

export async function GET() {
  try {
    const response = await fetch(`${process.env.NEXT_PUBLIC_API_BASE_URL}/health`, {
      headers: {
        'X-API-Key': process.env.NEXT_PUBLIC_API_KEY || '',
        'Accept': 'application/json',
      },
      // @ts-ignore
      next: {
        revalidate: 30
      }
    });

    if (!response.ok) {
      throw new Error(`Server responded with status: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Health check error:', error);
    return NextResponse.json(
      { error: 'Failed to check server health', details: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
} 