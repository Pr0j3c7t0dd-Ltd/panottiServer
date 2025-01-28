import { NextResponse } from 'next/server';

export async function POST() {
  try {
    // Send restart request to FastAPI server
    const res = await fetch('http://localhost:8000/api/restart', {
      method: 'POST',
    });

    if (!res.ok) {
      throw new Error('Failed to restart server');
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Failed to restart server:', error);
    return NextResponse.json(
      { success: false, message: 'Failed to restart server' },
      { status: 500 }
    );
  }
} 