'use client';

import { ShieldCheckIcon, CpuChipIcon, MicrophoneIcon, CodeBracketIcon } from '@heroicons/react/24/outline';

const features = [
  {
    title: 'Privacy First',
    description: 'Audio is captured and processed entirely on your device, ensuring your data stays private.',
    icon: ShieldCheckIcon,
  },
  {
    title: 'AI-Powered Workflows',
    description: 'Enable custom workflows for video conferencing, transcription, and more.',
    icon: CpuChipIcon,
  },
  {
    title: 'Flexibility',
    description: 'Easily send audio to any backend system using callbacks.',
    icon: MicrophoneIcon,
  },
  {
    title: 'Open Source',
    description: 'Powered by PanottiServer, an open-source API server for audio processing.',
    icon: CodeBracketIcon,
  },
];

export default function Features() {
  return (
    <section id="features" className="section-padding relative">
      <div className="gradient-blur pointer-events-none absolute inset-0 opacity-50" />
      <div className="relative mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="text-base font-semibold leading-7 text-blue-400">Features</h2>
          <p className="mt-2 text-3xl font-bold tracking-tight text-white sm:text-4xl">
            Everything you need for audio processing
          </p>
        </div>
        <div className="mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-none">
          <dl className="grid max-w-xl grid-cols-1 gap-x-8 gap-y-16 lg:max-w-none lg:grid-cols-2">
            {features.map((feature) => (
              <div key={feature.title} className="group relative">
                <div className="glass-card h-full p-8 transition-all duration-300 hover:bg-white/10">
                  <dt className="flex items-center gap-x-3 text-xl font-semibold leading-7 text-white">
                    <div className="rounded-lg bg-blue-600/10 p-3 ring-1 ring-blue-600/25 transition-colors group-hover:bg-blue-600/20">
                      <feature.icon className="size-6 text-blue-400" aria-hidden="true" />
                    </div>
                    {feature.title}
                  </dt>
                  <dd className="mt-4 text-base leading-7 text-zinc-400">
                    {feature.description}
                  </dd>
                </div>
              </div>
            ))}
          </dl>
        </div>
      </div>
    </section>
  );
}
