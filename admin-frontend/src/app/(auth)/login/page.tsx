import { LoginForm } from '@/components/auth/LoginForm';

export default function LoginPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[#030712]">
      <div className="gradient-blur pointer-events-none absolute inset-0" />
      <div className="relative z-10 w-full max-w-md px-6">
        <div className="glass-card p-8 space-y-8">
          <div>
            <h2 className="text-center text-3xl font-bold tracking-tight text-white">
              PanottiServer <br /> Admin Login
            </h2>
          </div>
          <LoginForm />
        </div>
      </div>
    </div>
  );
} 