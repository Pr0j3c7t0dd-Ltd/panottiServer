'use client';

export default function VideoSection() {
  return (
    <section id="video" className="relative py-16">
      <div className="gradient-blur pointer-events-none absolute inset-0 opacity-30" />
      <div className="relative mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-3xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
            See Panotti in Action
          </h2>
          <p className="mt-4 text-lg text-zinc-400">
            Watch how Panotti can transform your audio workflow
          </p>
        </div>
        <div className="relative mx-auto pt-4 text-center">
          <div className="">
            coming soon...
          </div>
          {/* <div className="aspect-w-16 aspect-h-9 relative overflow-hidden rounded-xl bg-zinc-900/50 shadow-xl">
            <iframe
              className="absolute inset-0 size-full"
              src="https://www.youtube.com/embed/dQw4w9WgXcQ"
              title="Panotti Introduction"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            />
          </div>*/}
        </div>
      </div>
    </section>
  );
}
