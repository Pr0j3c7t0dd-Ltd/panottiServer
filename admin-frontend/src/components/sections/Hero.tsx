'use client';

import Image from 'next/image';
import Link from 'next/link';
import { Button } from '@/components/ui/Button';

interface HeroProps {
  smoothScrollTo: (id: string) => void;
}

export default function Hero({ smoothScrollTo }: HeroProps) {
  return (
    <section id="hero" className="relative flex min-h-screen items-center">
      {/* Background effects */}
      <div className="gradient-blur pointer-events-none absolute inset-0" />
      <div className="absolute inset-x-0 -top-40 -z-10 transform-gpu overflow-hidden blur-3xl sm:-top-80">
        <div className="relative left-[calc(50%-11rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 rotate-[30deg] bg-gradient-to-tr from-[#1E3A8A] to-[#3B82F6] opacity-20 sm:left-[calc(50%-30rem)] sm:w-[72.1875rem]" />
      </div>

      <div className="mx-auto max-w-7xl px-6 pt-24 sm:pt-32 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <div className="float-animation relative mx-auto mb-12 size-32">
            <div className="absolute inset-0 rounded-full bg-blue-500/20 blur-2xl" />
            <Image
              src="/images/icon_final.webp"
              alt="Panotti Logo"
              width={256}
              height={256}
              className="relative mx-auto drop-shadow-2xl"
              loading="lazy"
              sizes="256px"
            />
          </div>
          <h1 className="text-5xl font-bold tracking-tight text-white sm:text-7xl">
            Capture, Process,{' '}
            <span className="text-gradient">Innovate</span>{' '}
            with Panotti
          </h1>
          <p className="mt-6 text-lg leading-8 text-zinc-400">
            Your private audio assistant. Supercharge your AI-powered workflows today.
          </p>
          <div className="mt-6 flex items-center justify-center">
            <span className="text-2xl font-bold text-zinc-400 line-through">$4.99</span>
            <span className="ml-2 text-3xl font-bold text-blue-400">FREE</span>
            <span className="ml-2 text-sm text-zinc-500">(Limited time offer until Feb 29)</span>
          </div>
          <div className="mt-10 flex flex-col items-center gap-y-6">
            <div className="flex items-center gap-x-6">
              <div className="rounded-md shadow">
                <a
                  href="https://apps.apple.com/app/id6739361057"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Button size="lg" className="w-full sm:w-auto">
                    Download on the Mac App Store
                  </Button>
                </a>
              </div>
              <Link
                href="#features"
                onClick={(e) => {
                  e.preventDefault();
                  smoothScrollTo('features');
                }}
                className="text-sm font-semibold text-zinc-400 transition-colors hover:text-white"
              >
                Learn More <span aria-hidden="true">→</span>
              </Link>
            </div>
            <a
              href="https://github.com/Pr0j3c7t0dd-Ltd/panottiServer"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-sm font-semibold text-zinc-400 transition-colors hover:text-white"
            >
              <svg className="size-5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
              </svg>
              View PanottiServer on GitHub <span aria-hidden="true">→</span>
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}
