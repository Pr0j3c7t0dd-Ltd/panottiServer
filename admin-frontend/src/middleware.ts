import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export async function middleware(request: NextRequest) {
  const response = NextResponse.next();

  // Add security headers
  response.headers.set('X-DNS-Prefetch-Control', 'on');
  response.headers.set('Strict-Transport-Security', 'max-age=31536000; includeSubDomains');
  response.headers.set('X-Frame-Options', 'SAMEORIGIN');
  response.headers.set('X-Content-Type-Options', 'nosniff');
  response.headers.set('X-XSS-Protection', '1; mode=block');
  response.headers.set('Referrer-Policy', 'strict-origin-when-cross-origin');
  response.headers.set('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');

  const isLoginPage = request.nextUrl.pathname === '/login';
  const isChangePasswordPage = request.nextUrl.pathname === '/admin/change-password';
  const isAuthCheckEndpoint = request.nextUrl.pathname.startsWith('/api/auth/');
  const sessionToken = request.cookies.get('session_token');

  // Allow access to login page if not authenticated
  if (isLoginPage) {
    if (sessionToken) {
      return NextResponse.redirect(new URL('/', request.url));
    }
    return response;
  }

  // Allow access to auth check endpoints
  if (isAuthCheckEndpoint) {
    return response;
  }

  // Protect all other routes
  if (!sessionToken) {
    return NextResponse.redirect(new URL('/login', request.url));
  }

  // If using default password, only allow access to change password page
  try {
    const baseUrl = request.nextUrl.origin;
    const checkDefaultRes = await fetch(`${baseUrl}/api/auth/check-default`);
    const { isDefault } = await checkDefaultRes.json();
    
    if (isDefault && !isChangePasswordPage) {
      return NextResponse.redirect(new URL('/admin/change-password', request.url));
    }
  } catch (error) {
    console.error('Failed to check default password:', error);
  }

  return response;
}

export const config = {
  matcher: [
    '/((?!api|_next/static|_next/image|favicon.ico).*)',
  ],
}; 