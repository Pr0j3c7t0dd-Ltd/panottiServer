'use client';

import { ChangePasswordModal } from '@/components/auth/ChangePasswordModal';
import { useRouter } from 'next/navigation';

export default function ChangePasswordPage() {
  const router = useRouter();

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#030712]">
      <div className="gradient-blur pointer-events-none absolute inset-0" />
      <ChangePasswordModal
        isOpen={true}
        onClose={() => router.push('/')}
        isDefault={true}
      />
    </div>
  );
} 