import React from 'react';
import Link from 'next/link';
import Header from '@/components/Header';
import Footer from '@/components/Footer';
import DiscordCommunity from '@/components/features/DiscordCommunity';

export default function DocsPage() {
  return (
    <div className="min-h-screen bg-black">
      <Header />
      <main className="relative isolate">
        {/* Background effects */}
        <div className="gradient-blur absolute inset-0" />
        <div className="absolute inset-x-0 -top-40 -z-10 transform-gpu overflow-hidden blur-3xl sm:-top-80">
          <div className="relative left-[calc(50%-11rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 rotate-[30deg] bg-gradient-to-tr from-[#1E3A8A] to-[#3B82F6] opacity-20 sm:left-[calc(50%-30rem)] sm:w-[72.1875rem]" />
        </div>

        {/* Hero section */}
        <div className="relative pt-14">
          <div className="py-24 sm:py-32">
            <div className="mx-auto max-w-7xl px-6 lg:px-8">
              <div className="mx-auto max-w-2xl text-center">
                <h1 className="text-4xl font-bold tracking-tight text-white sm:text-6xl">
                  Panotti Documentation
                </h1>
                <p className="mt-6 text-lg leading-8 text-zinc-400">
                  Everything you need to know about using Panotti
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Documentation cards */}
        <div className="mx-auto max-w-7xl px-6 pb-24 lg:px-8">
          <div className="mx-auto max-w-2xl lg:max-w-4xl">
            <div className="grid grid-cols-1 gap-8 sm:grid-cols-1 lg:grid-cols-2">
              {/* Getting Started Card */}
              <Link href="/docs/getting-started" className="group relative isolate sm:col-span-1 lg:col-span-2 mx-auto flex max-w-2xl flex-col justify-between rounded-2xl border border-white/10 bg-gray-900 px-6 pb-6 pt-8 transition-colors hover:border-white/20 sm:pt-10">
                <div>
                  <h3 className="text-2xl font-semibold leading-7 text-white">
                    Getting Started
                  </h3>
                  <p className="mt-4 text-base leading-7 text-zinc-400">
                    New to Panotti? Start here to learn the basics and get up and running quickly.
                  </p>
                </div>
                <div className="mt-8 flex items-center gap-x-2 text-sm font-semibold leading-6 text-white">
                  View documentation
                  <svg className="size-5 transition-transform group-hover:translate-x-1" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M3 10a.75.75 0 01.75-.75h10.638L10.23 5.29a.75.75 0 111.04-1.08l5.5 5.25a.75.75 0 010 1.08l-5.5 5.25a.75.75 0 11-1.04-1.08l4.158-3.96H3.75A.75.75 0 013 10z" clipRule="evenodd" />
                  </svg>
                </div>
              </Link>

              {/* App Documentation Card */}
              <Link href="/docs/app" className="group relative isolate flex flex-col justify-between rounded-2xl border border-white/10 bg-gray-900 px-6 pb-6 pt-8 transition-colors hover:border-white/20 sm:pt-10">
                <div>
                  <h3 className="text-2xl font-semibold leading-7 text-white">
                    App Documentation
                  </h3>
                  <p className="mt-4 text-base leading-7 text-zinc-400">
                    Learn how to use the Panotti app to manage your audio devices, set up shortcuts, and customize your experience.
                  </p>
                </div>
                <div className="mt-8 flex items-center gap-x-2 text-sm font-semibold leading-6 text-white">
                  View documentation
                  <svg className="size-5 transition-transform group-hover:translate-x-1" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M3 10a.75.75 0 01.75-.75h10.638L10.23 5.29a.75.75 0 111.04-1.08l5.5 5.25a.75.75 0 010 1.08l-5.5 5.25a.75.75 0 11-1.04-1.08l4.158-3.96H3.75A.75.75 0 013 10z" clipRule="evenodd" />
                  </svg>
                </div>
              </Link>

              {/* Server Documentation Card */}
              <Link href="/docs/server" className="group relative isolate flex flex-col justify-between rounded-2xl border border-white/10 bg-gray-900 px-6 pb-6 pt-8 transition-colors hover:border-white/20 sm:pt-10">
                <div>
                  <h3 className="text-2xl font-semibold leading-7 text-white">
                    Server Documentation
                  </h3>
                  <p className="mt-4 text-base leading-7 text-zinc-400">
                    Set up and manage your Panotti server instance, configure integrations, and understand the API.
                  </p>
                </div>
                <div className="mt-8 flex items-center gap-x-2 text-sm font-semibold leading-6 text-white">
                  View documentation
                  <svg className="size-5 transition-transform group-hover:translate-x-1" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M3 10a.75.75 0 01.75-.75h10.638L10.23 5.29a.75.75 0 111.04-1.08l5.5 5.25a.75.75 0 010 1.08l-5.5 5.25a.75.75 0 11-1.04-1.08l4.158-3.96H3.75A.75.75 0 013 10z" clipRule="evenodd" />
                  </svg>
                </div>
              </Link>
            </div>

            {/* Discord Community Section */}
            <DiscordCommunity className="mt-16" />
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}
