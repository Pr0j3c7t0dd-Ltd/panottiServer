'use client';

import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';

const ServerStatus = () => {
  const { data, isError, error } = useQuery({
    queryKey: ['serverHealth'],
    queryFn: async () => {
      try {
        // Call our Next.js API route instead of the server directly
        const response = await fetch('/api/health');
        
        if (!response.ok) {
          const errorData = await response.text();
          throw new Error(
            `Server error (${response.status}): ${errorData}`
          );
        }
        return response.json();
      } catch (err) {
        if (err instanceof Error) {
          // Check for SSL certificate error and provide more helpful message in development
          if (err.message.includes('SSL') || err.message.includes('certificate') || err.message.includes('ERR_CERT_AUTHORITY_INVALID')) {
            if (process.env.NODE_ENV === 'development') {
              throw new Error(
                'SSL Certificate Error: You are using a self-signed certificate in development.\n\n' +
                'To fix this, you need to:\n' +
                '1. Open https://localhost:54789/health directly in your browser\n' +
                '2. Click "Advanced"\n' +
                '3. Click "Proceed to localhost (unsafe)"\n' +
                '4. Return to this page and refresh'
              );
            }
            throw new Error('SSL Certificate Error: Invalid certificate');
          }
          throw new Error(`Failed to check server status: ${err.message}`);
        }
        throw new Error('An unknown error occurred');
      }
    },
    refetchInterval: 30000, // Poll every 30 seconds
    retry: 3, // Retry failed requests 3 times
    refetchOnWindowFocus: false, // Don't refetch on window focus
    refetchOnMount: false, // Don't refetch on mount
    staleTime: 30000, // Consider data fresh for 30 seconds
  });

  return (
    <div id="server-status" className="p-6 rounded-lg shadow-md text-center">
      <h2 className="text-xl font-semibold mb-4">Server Status</h2>
      <div
        className={`p-8 rounded-lg ${
          isError ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
        }`}
      >
        <p className="text-3xl font-bold">{isError ? 'DOWN' : 'UP'}</p>
        {isError && error instanceof Error && (
          <p className="mt-2 text-sm whitespace-pre-wrap">{error.message}</p>
        )}
      </div>
    </div>
  );
};

export default ServerStatus; 