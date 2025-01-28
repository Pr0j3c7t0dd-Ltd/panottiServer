'use client';

import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';

const ServerStatus = () => {
  const { data, isError, error } = useQuery({
    queryKey: ['serverHealth'],
    queryFn: async () => {
      try {
        const response = await fetch(`${process.env.NEXT_PUBLIC_API_BASE_URL}/health`, {
          headers: {
            'X-API-Key': process.env.NEXT_PUBLIC_API_KEY || '',
            'Accept': 'application/json',
          },
          // Required for self-signed certificates in development
          mode: 'cors',
        });
        
        if (!response.ok) {
          const errorData = await response.text();
          throw new Error(
            `Server error (${response.status}): ${errorData}`
          );
        }
        return response.json();
      } catch (err) {
        if (err instanceof Error) {
          // Check for SSL certificate error
          if (err.message.includes('SSL') || err.message.includes('certificate')) {
            throw new Error('SSL Certificate Error: The server is using a self-signed certificate. Please ensure you trust the certificate.');
          }
          throw err;
        }
        throw new Error('An unknown error occurred');
      }
    },
    refetchInterval: 30000, // Poll every 30 seconds
    retry: 3, // Retry failed requests 3 times
  });

  return (
    <div className="p-6 rounded-lg shadow-md text-center">
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