'use client';

import { useEffect } from 'react';

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Check for chunk loading errors and reload
    if (error.message.includes('Loading chunk')) {
      window.location.reload();
      return;
    }
    // Log other errors to an error reporting service
    console.error(error);
  }, [error]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#030712]">
      <div className="gradient-blur pointer-events-none absolute inset-0" />
      <div className="relative z-10 mx-auto max-w-7xl px-6 text-center">
        <p className="text-base font-semibold text-red-600">Error</p>
        <h1 className="mt-4 text-3xl font-bold tracking-tight text-white sm:text-5xl">Something went wrong!</h1>
        <p className="mt-6 text-base leading-7 text-zinc-400">
          {error.message || 'An unexpected error occurred'}
        </p>
        <div className="mt-10 flex items-center justify-center gap-x-6">
          <button
            onClick={reset}
            className="rounded-md bg-blue-600 px-3.5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-blue-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600"
          >
            Try again
          </button>
          <a
            href="/"
            className="text-sm font-semibold text-zinc-400 hover:text-zinc-300"
          >
            Go back home
          </a>
        </div>
      </div>
    </div>
  );
} 