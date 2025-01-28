'use client';

import React from 'react';
import Header from './Header';
import Footer from './Footer';
import ErrorBoundary from '../error/ErrorBoundary';
import PageTransition from '../ui/PageTransition';
import { useToast } from '../../hooks/useToast';

interface MainLayoutProps {
  children: React.ReactNode;
}

const MainLayout = ({ children }: MainLayoutProps) => {
  const { ToastComponent } = useToast();

  return (
    <ErrorBoundary>
      <div className="flex min-h-screen flex-col">
        <Header />
        <PageTransition>
          <main className="flex-1">{children}</main>
        </PageTransition>
        <Footer />
        {ToastComponent}
      </div>
    </ErrorBoundary>
  );
};

export default MainLayout;
