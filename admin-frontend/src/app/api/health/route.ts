import { NextResponse } from 'next/server';

export async function GET() {
  // Extract host from API base URL
  const apiUrl = new URL(process.env.NEXT_PUBLIC_API_BASE_URL || '');
  
  try {
    // Try to resolve the hostname first with IPv4
    try {
      const dnsPromise = await import('dns/promises');
      const addresses = await dnsPromise.lookup(apiUrl.hostname, { family: 4 });
      
      // Use the resolved IPv4 address
      const ipv4Url = new URL(process.env.NEXT_PUBLIC_API_BASE_URL || '');
      ipv4Url.hostname = addresses.address;
      
      // Construct the health check URL properly
      const healthCheckUrl = new URL('/health', ipv4Url);
      
      const response = await fetch(healthCheckUrl.toString(), {
        headers: {
          'X-API-Key': process.env.NEXT_PUBLIC_API_KEY || '',
          'Accept': 'application/json',
          'Host': apiUrl.host
        },
        cache: 'no-store'
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server responded with status: ${response.status}, body: ${errorText}`);
      }

      const data = await response.json();
      return NextResponse.json(data);
      
    } catch (dnsError) {
      throw dnsError;
    }
  } catch (error) {
    const errorMessage = error instanceof Error 
      ? `${error.message}${error.cause ? `\nCause: ${JSON.stringify(error.cause, null, 2)}` : ''}`
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