import bcrypt from 'bcryptjs';
import fs from 'fs/promises';
import path from 'path';

const HASH_FILE = path.join(process.cwd(), 'password-hash.txt');
const DEFAULT_PASSWORD = 'Pa55w0rd';
const SALT_ROUNDS = 10;

export async function initializePasswordHash() {
  try {
    await fs.access(HASH_FILE);
  } catch {
    const hash = await bcrypt.hash(DEFAULT_PASSWORD, SALT_ROUNDS);
    await fs.writeFile(HASH_FILE, hash);
  }
}

export async function verifyPassword(password: string): Promise<boolean> {
  const hash = await fs.readFile(HASH_FILE, 'utf-8');
  return bcrypt.compare(password, hash);
}

export async function changePassword(oldPassword: string, newPassword: string): Promise<boolean> {
  const isValid = await verifyPassword(oldPassword);
  if (!isValid) return false;

  const hash = await bcrypt.hash(newPassword, SALT_ROUNDS);
  await fs.writeFile(HASH_FILE, hash);
  return true;
}

export async function isDefaultPassword(): Promise<boolean> {
  return verifyPassword(DEFAULT_PASSWORD);
} 