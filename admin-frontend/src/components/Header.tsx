'use client';

import Link from 'next/link';
import Image from 'next/image';
import { usePathname, useRouter } from 'next/navigation';
import { useState } from 'react';
import AnnouncementStrip from '@/components/ui/AnnouncementStrip';
import { Button } from '@/components/ui/Button';
import { ChangePasswordModal } from '@/components/auth/ChangePasswordModal';

export default function Header() {
  const pathname = usePathname();
  const router = useRouter();
  const isHomePage = pathname === '/';
  const [showChangePassword, setShowChangePassword] = useState(false);

  const handleNavClick = (e: React.MouseEvent<HTMLAnchorElement>, id: string) => {
    if (isHomePage) {
      e.preventDefault();
      const element = document.getElementById(id);
      if (element) {
        const headerOffset = 112; // Increased to account for announcement strip
        const elementPosition = element.getBoundingClientRect().top;
        const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

        window.scrollTo({
          top: offsetPosition,
          behavior: 'smooth'
        });
      }
    }
  };

  const handleLogout = async () => {
    try {
      const res = await fetch('/api/logout', {
        method: 'POST',
      });

      if (!res.ok) throw new Error('Failed to logout');

      router.push('/login');
    } catch (err) {
      console.error('Failed to logout:', err);
    }
  };

  return (
    <div className="absolute inset-x-0 top-0">
      <div className="fixed inset-x-0 top-0 z-50">
        <AnnouncementStrip />
        <header className="border-b border-white/10 bg-black/50 backdrop-blur-xl">
          <nav className="mx-auto flex max-w-7xl items-center justify-between p-6 lg:px-8" aria-label="Global">
            <div className="flex lg:flex-1">
              <Link href="/" className="-m-1.5 p-1.5 transition-transform hover:scale-105">
                <span className="sr-only">Panotti</span>
                <div className="flex items-center gap-3">
                  <Image
                    src="/images/icon_final.webp"
                    alt="Panotti Logo"
                    width={64}
                    height={64}
                    className="size-8"
                    priority
                    sizes="64px"
                  />
                  <span className="text-lg font-semibold text-white">PanottiServer Admin</span>
                </div>
              </Link>
            </div>
            <div className="flex items-center gap-x-4 sm:gap-x-8 lg:gap-x-12">
              <Link
                href={isHomePage ? "#" : "/"}
                className="text-sm font-medium text-zinc-400 transition-colors hover:text-white"
              >
                Home
              </Link>
              <Link
                href={isHomePage ? "#contact" : "/#contact"}
                onClick={(e) => handleNavClick(e, 'contact')}
                className="text-sm font-medium text-zinc-400 transition-colors hover:text-white"
              >
                Contact Us
              </Link>
              <button
                onClick={() => setShowChangePassword(true)}
                className="text-sm font-medium text-zinc-400 transition-colors hover:text-white"
              >
                Change Password
              </button>
              <Button
                variant="secondary"
                onClick={handleLogout}
                size="sm"
              >
                Logout
              </Button>
            </div>
          </nav>
        </header>
      </div>
      <ChangePasswordModal
        isOpen={showChangePassword}
        onClose={() => setShowChangePassword(false)}
      />
    </div>
  );
}
