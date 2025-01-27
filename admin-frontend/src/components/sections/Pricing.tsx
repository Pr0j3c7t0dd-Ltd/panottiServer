'use client';

import { SparklesIcon, RocketLaunchIcon, StarIcon } from '@heroicons/react/24/outline';

const tiers = [
  {
    title: 'Free',
    price: '$0',
    description: 'Perfect for getting started and exploring our features.',
    features: ['Basic audio processing', 'Community support', 'Standard quality'],
    icon: SparklesIcon,
  },
  {
    title: 'Normal',
    price: '$4.99',
    period: '/month',
    description: 'Great for individual developers and small teams.',
    features: ['Advanced audio processing', 'Priority support', 'High quality', 'Custom workflows'],
    icon: RocketLaunchIcon,
  },
  {
    title: 'Pro',
    price: '$10',
    period: '/month',
    description: 'For teams that need the best in audio processing.',
    features: ['Enterprise audio processing', '24/7 support', 'Ultra high quality', 'Custom workflows', 'API access'],
    icon: StarIcon,
  },
];

export default function Pricing() {
  return (
    <section id="pricing" className="section-padding relative">
      <div className="gradient-blur pointer-events-none absolute inset-0 opacity-50" />
      <div className="relative mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="text-base font-semibold leading-7 text-blue-400">Pricing</h2>
          <p className="mt-2 text-3xl font-bold tracking-tight text-white sm:text-4xl">
            Choose the plan that's right for you
          </p>
        </div>
        <div className="mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-none">
          <div className="grid max-w-xl grid-cols-1 gap-x-8 gap-y-16 lg:max-w-none lg:grid-cols-3">
            {tiers.map((tier) => (
              <div key={tier.title} className="group relative">
                <div className="glass-card h-full p-8 transition-all duration-300 hover:bg-white/10">
                  <dt className="flex items-center gap-x-3 text-xl font-semibold leading-7 text-white">
                    <div className="rounded-lg bg-blue-600/10 p-3 ring-1 ring-blue-600/25 transition-colors group-hover:bg-blue-600/20">
                      <tier.icon className="size-6 text-blue-400" aria-hidden="true" />
                    </div>
                    {tier.title}
                  </dt>
                  <dd className="mt-4 flex flex-col gap-4">
                    <div className="flex items-baseline">
                      <span className="text-4xl font-bold text-white">{tier.price}</span>
                      {tier.period && <span className="text-xl text-zinc-400">{tier.period}</span>}
                    </div>
                    <p className="text-base leading-7 text-zinc-400">{tier.description}</p>
                    <ul className="mt-4 space-y-3">
                      {tier.features.map((feature) => (
                        <li key={feature} className="flex items-center text-zinc-400">
                          <CheckIcon className="mr-3 size-5 text-blue-400" />
                          {feature}
                        </li>
                      ))}
                    </ul>
                  </dd>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function CheckIcon(props: React.ComponentProps<'svg'>) {
  return (
    <svg
      {...props}
      fill="none"
      viewBox="0 0 24 24"
      strokeWidth={2}
      stroke="currentColor"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M4.5 12.75l6 6 9-13.5"
      />
    </svg>
  );
}
