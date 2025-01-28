'use client';

import Header from '@/components/Header';
import Footer from '@/components/Footer';
import { useSmoothScroll } from '@/hooks/useSmoothScroll';
import Contact from '@/components/sections/Contact';
import ServerStatus from '@/components/ui/ServerStatus';

export default function Home() {
  const smoothScrollTo = useSmoothScroll();

  return (
    <>
      <Header />
      <main className="overflow-hidden">
        <div className="container mx-auto px-4 mt-48 pb-4 sm:mt-36">
          <ServerStatus />
        </div>
        <Contact />
      </main>
      <Footer />
    </>
  );
}