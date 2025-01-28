import { NextResponse } from 'next/server';
import { changePassword } from '@/lib/auth';

export async function POST(request: Request) {
  try {
    const { oldPassword, newPassword } = await request.json();

    if (!oldPassword || !newPassword) {
      return NextResponse.json(
        { success: false, message: 'Both old and new passwords are required' },
        { status: 400 }
      );
    }

    const success = await changePassword(oldPassword, newPassword);

    if (!success) {
      return NextResponse.json(
        { success: false, message: 'Current password is incorrect' },
        { status: 401 }
      );
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Change password error:', error);
    return NextResponse.json(
      { success: false, message: 'Internal server error' },
      { status: 500 }
    );
  }
} 