import { NextResponse } from 'next/server';

export async function GET() {
  // Extract host from API base URL and add debugging
  const apiUrl = new URL(process.env.NEXT_PUBLIC_API_BASE_URL || '');
  
  try {
    console.log('Debug - API URL parts:', {
      full: process.env.NEXT_PUBLIC_API_BASE_URL,
      protocol: apiUrl.protocol,
      hostname: apiUrl.hostname,
      port: apiUrl.port,
      host: apiUrl.host,
      pathname: apiUrl.pathname
    });
    
    // Try to resolve the hostname first with IPv4
    try {
      const dnsPromise = await import('dns/promises');
      const addresses = await dnsPromise.lookup(apiUrl.hostname, { family: 4 });
      console.log('Debug - DNS lookup:', addresses);
      
      // Use the resolved IPv4 address
      const ipv4Url = new URL(process.env.NEXT_PUBLIC_API_BASE_URL || '');
      ipv4Url.hostname = addresses.address;
      
      // Construct the health check URL properly
      const healthCheckUrl = new URL('/health', ipv4Url);
      console.log('Debug - Using IPv4 URL:', healthCheckUrl.toString());
      
      console.log('Debug - Attempting fetch with headers:', {
        'X-API-Key': process.env.NEXT_PUBLIC_API_KEY ? '(set)' : '(not set)',
        'Accept': 'application/json',
        'Host': apiUrl.host
      });
      
      const response = await fetch(healthCheckUrl.toString(), {
        headers: {
          'X-API-Key': process.env.NEXT_PUBLIC_API_KEY || '',
          'Accept': 'application/json',
          'Host': apiUrl.host
        },
        // Required for self-signed certificates
        cache: 'no-store'
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server responded with status: ${response.status}, body: ${errorText}`);
      }

      const data = await response.json();
      console.log('Debug - Successful response:', data);
      return NextResponse.json(data);
      
    } catch (dnsError) {
      console.log('Debug - DNS lookup failed:', dnsError);
      throw dnsError;
    }
  } catch (error) {
    console.error('Health check error:', error);
    
    // More detailed error message for debugging
    const errorMessage = error instanceof Error 
      ? `${error.message}${error.cause ? `\nCause: ${JSON.stringify(error.cause, null, 2)}` : ''}`
      : 'Unknown error';
      
    console.log('Debug - Full error details:', {
      message: errorMessage,
      error: error instanceof Error ? {
        name: error.name,
        message: error.message,
        cause: error.cause,
        stack: error.stack
      } : error
    });
      
    return NextResponse.json(
      { 
        error: 'Failed to check server health', 
        details: errorMessage,
        url: process.env.NEXT_PUBLIC_API_BASE_URL,
        debug: {
          apiUrl: {
            full: process.env.NEXT_PUBLIC_API_BASE_URL,
            parsed: {
              protocol: apiUrl.protocol,
              hostname: apiUrl.hostname,
              port: apiUrl.port,
              host: apiUrl.host
            }
          }
        }
      },
      { status: 500 }
    );
  }
} 