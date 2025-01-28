'use client';

import { useCallback } from 'react';

export const useSmoothScroll = () => {
  const smoothScrollTo = useCallback((elementId: string) => {
    const element = document.getElementById(elementId);
    if (element) {
      element.scrollIntoView({
        behavior: 'smooth',
        block: 'start',
      });
    }
  }, []);

  return smoothScrollTo;
};
