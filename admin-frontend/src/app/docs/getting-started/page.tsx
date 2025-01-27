import React from 'react';
import Header from '@/components/Header';
import Footer from '@/components/Footer';
import DocsSideMenu from '@/components/features/docs/DocsSideMenu';
import DiscordCommunity from '@/components/features/DiscordCommunity';

export default function GettingStartedPage() {
  return (
    <div className="min-h-screen bg-black">
      <Header />
      <main className="relative isolate">
        {/* Background effects */}
        <div className="gradient-blur absolute inset-0" />
        <div className="absolute inset-x-0 -top-40 -z-10 transform-gpu overflow-hidden blur-3xl sm:-top-80">
          <div className="relative left-[calc(50%-11rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 rotate-[30deg] bg-gradient-to-tr from-[#1E3A8A] to-[#3B82F6] opacity-20 sm:left-[calc(50%-30rem)] sm:w-[72.1875rem]" />
        </div>

        <div className="relative pt-36">
          <div className="container mx-auto px-4 py-8">
            <div className="block lg:flex">
              <div className="mb-8 lg:hidden">
                <DocsSideMenu />
              </div>
              <div className="hidden lg:block">
                <DocsSideMenu />
              </div>
              <main className="max-w-3xl flex-1">
                <h1 className="mb-8 text-4xl font-bold">Getting Started with Panotti</h1>

                <section id="introduction">
                  <h2 className="mb-4 text-2xl font-semibold">Introduction</h2>
                  <p className="mb-4 text-zinc-300">
                    Welcome to Panotti! This guide will help you get up and running with our platform quickly and efficiently.
                  </p>
                </section>

                <section id="overview" className="mt-8">
                  <h2 className="mb-4 text-2xl font-semibold">Overview</h2>
                  <p className="mb-4 text-zinc-300">
                    Panotti consists of two main components that can work together or independently:
                  </p>

                  <div className="mb-6">
                    <h3 className="mb-2 text-xl font-semibold text-zinc-100">1. Panotti App</h3>
                    <p className="mb-4 text-zinc-300">
                      A powerful desktop application that captures both microphone and system audio. The app can trigger
                      webhook callbacks to any API endpoint at the start and end of recordings, making it highly flexible
                      for integration with your existing systems.
                    </p>
                  </div>

                  <div className="mb-6">
                    <h3 className="mb-2 text-xl font-semibold text-zinc-100">2. Backend Server</h3>
                    <p className="mb-4 text-zinc-300">
                      An open-source server that processes audio to generate transcripts and meeting notes. It features
                      a plugin architecture that allows you to create custom plugins to extend its functionality according
                      to your needs.
                    </p>
                  </div>

                  <div className="rounded-lg bg-zinc-900 p-4">
                    <p className="text-zinc-300">
                      While both components can be used independently, we recommend using them together for the best experience
                      and easiest setup. The integrated solution provides a seamless workflow from audio recording to
                      processed meeting insights.
                    </p>
                  </div>
                </section>

                <section id="prerequisites" className="mt-8">
                  <h2 className="mb-4 text-2xl font-semibold">Minimum Prerequisites</h2>
                  <ul className="list-disc pl-6 text-zinc-300">
                    <li className="mb-2">MacOS 15.2 (Sequoia)</li>
                  </ul>
                </section>

                <section id="quick-start" className="mt-8">
                  <h2 className="mb-4 text-2xl font-semibold">Quick Start</h2>
                  <div>coming soon...</div>
                  {/* <ol className="list-decimal pl-6 text-zinc-300">
                    <li className="mb-4">
                      <p className="font-semibold">Install the Panotti CLI</p>
                      <pre className="mt-2 rounded-md bg-zinc-900 p-4">
                        <code>npm install -g @panotti/cli</code>
                      </pre>
                    </li>
                    <li className="mb-4">
                      <p className="font-semibold">Create a new Panotti project</p>
                      <pre className="mt-2 rounded-md bg-zinc-900 p-4">
                        <code>panotti create my-project</code>
                      </pre>
                    </li>
                    <li className="mb-4">
                      <p className="font-semibold">Start the development server</p>
                      <pre className="mt-2 rounded-md bg-zinc-900 p-4">
                        <code>cd my-project
                          npm run dev</code>
                      </pre>
                    </li>
                  </ol> */}
                </section>

                <section id="next-steps" className="mt-8">
                  <h2 className="mb-4 text-2xl font-semibold">Next Steps</h2>
                  <p className="mb-4 text-zinc-300">
                    Now that you have Panotti set up, you can:
                  </p>
                  <ul className="list-disc pl-6 text-zinc-300">
                    <li className="mb-2">Explore the <a href="/docs/app" className="text-blue-400 hover:text-blue-300">App Documentation</a> for client-side features</li>
                    <li className="mb-2">Check out the <a href="/docs/server" className="text-blue-400 hover:text-blue-300">Server Documentation</a> for backend setup</li>
                    <li className="mb-2">Join our community on Discord for support</li>
                  </ul>
                </section>

                <div className="mt-8">
                  <DiscordCommunity />
                </div>
              </main>
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}
