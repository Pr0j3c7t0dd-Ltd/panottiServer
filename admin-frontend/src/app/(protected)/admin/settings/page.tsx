'use client';

import { useState } from 'react';
import { EnvironmentVariables } from '@/components/settings/EnvironmentVariables';
import { PluginSettings } from '@/components/settings/PluginSettings';
import { RestartModal } from '@/components/settings/RestartModal';
import Header from '@/components/Header';
import Footer from '@/components/Footer';

export default function SettingsPage() {
  const [showRestartModal, setShowRestartModal] = useState(false);
  const [restartReason, setRestartReason] = useState('');

  const handleRestart = (reason: string) => {
    setRestartReason(reason);
    setShowRestartModal(true);
  };

  return (
    <>
      <Header />
      <main className="min-h-screen bg-[#030712]">
        <div className="gradient-blur pointer-events-none absolute inset-0" />
        <div className="relative z-10 mx-auto max-w-7xl px-6 pt-40 pb-4 lg:px-8">
          <div className="space-y-8">
            <div className="rounded-2xl border border-white/10 bg-[#030712] p-8">
              <h1 className="text-3xl font-bold tracking-tight text-white">Settings</h1>
              <p className="mt-2 text-zinc-400">
                Manage your environment variables and plugin settings.
              </p>
            </div>

            <div className="space-y-8">
              <div className="rounded-2xl border border-white/10 bg-[#030712] p-8">
                <EnvironmentVariables onRestart={handleRestart} />
              </div>
              <div className="rounded-2xl border border-white/10 bg-[#030712] p-8">
                <PluginSettings onRestart={handleRestart} />
              </div>
            </div>
          </div>
        </div>
      </main>
      <Footer />
      <RestartModal
        isOpen={showRestartModal}
        onClose={() => setShowRestartModal(false)}
        onConfirm={() => setShowRestartModal(false)}
        reason={restartReason}
      />
    </>
  );
} 