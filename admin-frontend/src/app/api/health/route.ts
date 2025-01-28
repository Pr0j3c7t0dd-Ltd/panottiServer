import { NextResponse } from 'next/server';

export async function GET() {
  // Extract host from API base URL
  const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
  if (!apiBaseUrl) {
    return NextResponse.json(
      { error: 'API base URL not configured' },
      { status: 500 }
    );
  }

  try {
    // Remove trailing slash if present and add health endpoint
    const baseUrl = apiBaseUrl.endsWith('/') ? apiBaseUrl.slice(0, -1) : apiBaseUrl;
    const healthCheckUrl = `${baseUrl}/health`;

    const response = await fetch(healthCheckUrl, {
      headers: {
        'X-API-Key': process.env.NEXT_PUBLIC_API_KEY || '',
        'Accept': 'application/json',
      },
      cache: 'no-store'
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`Health check failed: ${response.status} - ${errorText}`);
      return NextResponse.json(
        { error: 'Health check failed', status: response.status, details: errorText },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
    
  } catch (error) {
    const errorMessage = error instanceof Error 
      ? error.message
      : 'Unknown error occurred';
      
    console.error('Health check error:', errorMessage);
    
    return NextResponse.json(
      { 
        error: 'Failed to check server health', 
        details: errorMessage,
        url: apiBaseUrl
      },
      { status: 500 }
    );
  }
} 