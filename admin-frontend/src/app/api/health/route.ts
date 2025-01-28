import { NextResponse } from 'next/server';
import axios, { AxiosError } from 'axios';
import https from 'https';

// This is needed for self-signed certificates
process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

export async function GET() {
  const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
  if (!apiBaseUrl) {
    return NextResponse.json(
      { error: 'API base URL not configured' },
      { status: 500 }
    );
  }

  try {
    const url = new URL(apiBaseUrl);
    
    // Force IPv4 for localhost
    if (url.hostname === 'localhost') {
      url.hostname = '127.0.0.1';
    }
    
    const healthCheckUrl = new URL('health', url).toString();

    const httpsAgent = new https.Agent({
      rejectUnauthorized: false
    });

    const response = await axios.get(healthCheckUrl, {
      httpsAgent,
      headers: {
        'X-API-Key': process.env.NEXT_PUBLIC_API_KEY || '',
        'Accept': 'application/json'
      }
    });

    return NextResponse.json(response.data);
    
  } catch (error) {
    console.error('Health check error:', error);

    if (error instanceof AxiosError) {
      return NextResponse.json(
        {
          error: 'Failed to check server health',
          details: error.response?.data || error.message,
          status: error.response?.status
        },
        { status: error.response?.status || 500 }
      );
    }
    
    return NextResponse.json(
      { error: 'Failed to check server health' },
      { status: 500 }
    );
  }
} 