'use client';

import DiscordCommunity from '@/components/features/DiscordCommunity';

export default function Contact() {
  return (
    <section id="contact" className="relative mt-32 sm:mt-40">
      <div className="gradient-blur pointer-events-none absolute inset-0 opacity-30" />
      <div className="relative mx-auto max-w-7xl px-6 py-24 sm:py-32 lg:px-8">
        <div className="mx-auto max-w-3xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">Get in Touch</h2>
          <p className="mt-6 text-lg leading-8 text-zinc-400">
            We value your feedback and are always looking to improve. Whether you have suggestions,
            feature requests, or are interested in exploring business opportunities and partnerships,
            we'd love to hear from you.
          </p>
          <div className="relative z-20 mb-12 mt-8 text-2xl leading-relaxed text-zinc-400 sm:text-3xl">
            Drop us a line at{' '}
            <a
              href="mailto:hello@panotti.io"
              className="relative inline-block cursor-pointer font-semibold text-blue-400 transition-colors hover:text-blue-300 hover:underline"
            >
              hello@panotti.io
            </a>
          </div>
          <div className="mx-auto max-w-xl">
            <DiscordCommunity className="mt-8" />
          </div>
        </div>
      </div>
    </section>
  );
}
