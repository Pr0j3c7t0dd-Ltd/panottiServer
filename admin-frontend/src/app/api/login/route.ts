import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { verifyPassword, isDefaultPassword } from '@/lib/auth';

export async function POST(request: Request) {
  try {
    const { password } = await request.json();

    const isValid = await verifyPassword(password);
    if (!isValid) {
      return NextResponse.json(
        { success: false, message: 'Invalid password' },
        { status: 401 }
      );
    }

    // Check if using default password
    const isDefault = await isDefaultPassword();

    // Set session cookie
    const cookieStore = cookies();
    cookieStore.set('session_token', 'authenticated', {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 60 * 60 * 24, // 24 hours
    });

    return NextResponse.json({
      success: true,
      requiresPasswordChange: isDefault,
    });
  } catch (error) {
    console.error('Login error:', error);
    return NextResponse.json(
      { success: false, message: 'Internal server error' },
      { status: 500 }
    );
  }
} 