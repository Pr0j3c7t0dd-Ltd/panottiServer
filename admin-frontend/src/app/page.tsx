'use client';

import Header from '@/components/Header';
import Footer from '@/components/Footer';
import { useSmoothScroll } from '@/hooks/useSmoothScroll';
import Hero from '@/components/sections/Hero';
import VideoSection from '@/components/sections/VideoSection';
import Features from '@/components/sections/Features';
import Contact from '@/components/sections/Contact';

export default function Home() {
  const smoothScrollTo = useSmoothScroll();

  return (
    <>
      <Header />
      <main className="overflow-hidden">
        <Hero smoothScrollTo={smoothScrollTo} />
        <VideoSection />
        <Features />
        <Contact />
      </main>
      <Footer />
    </>
  );
}