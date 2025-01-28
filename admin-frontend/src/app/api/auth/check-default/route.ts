import { NextResponse } from 'next/server';
import { isDefaultPassword } from '@/lib/auth';

export async function GET() {
  try {
    const isDefault = await isDefaultPassword();
    return NextResponse.json({ isDefault });
  } catch (error) {
    console.error('Check default password error:', error);
    return NextResponse.json(
      { success: false, message: 'Internal server error' },
      { status: 500 }
    );
  }
} 