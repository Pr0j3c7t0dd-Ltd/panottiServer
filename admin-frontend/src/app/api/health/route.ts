import { NextResponse } from 'next/server';

export async function GET() {
  try {
    const response = await fetch(`${process.env.NEXT_PUBLIC_API_BASE_URL}/health`, {
      headers: {
        'X-API-Key': process.env.NEXT_PUBLIC_API_KEY || '',
        'Accept': 'application/json',
      },
      // Required for self-signed certificates
      cache: 'no-store'
    });

    if (!response.ok) {
      throw new Error(`Server responded with status: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Health check error:', error);
    
    // More detailed error message for debugging
    const errorMessage = error instanceof Error 
      ? `${error.message}${error.cause ? ` (Cause: ${error.cause})` : ''}`
      : 'Unknown error';
      
    return NextResponse.json(
      { 
        error: 'Failed to check server health', 
        details: errorMessage,
        url: process.env.NEXT_PUBLIC_API_BASE_URL 
      },
      { status: 500 }
    );
  }
} 