export default function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[#030712]">
      <div className="gradient-blur pointer-events-none absolute inset-0" />
      <div className="relative z-10 mx-auto max-w-7xl px-6 text-center">
        <p className="text-base font-semibold text-blue-600">404</p>
        <h1 className="mt-4 text-3xl font-bold tracking-tight text-white sm:text-5xl">Page not found</h1>
        <p className="mt-6 text-base leading-7 text-zinc-400">Sorry, we couldn't find the page you're looking for.</p>
        <div className="mt-10 flex items-center justify-center gap-x-6">
          <a
            href="/"
            className="rounded-md bg-blue-600 px-3.5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-blue-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600"
          >
            Go back home
          </a>
        </div>
      </div>
    </div>
  );
} 