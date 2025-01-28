const bcrypt = require('bcryptjs');
const fs = require('fs/promises');
const path = require('path');

const HASH_FILE = path.join(process.cwd(), 'password-hash.txt');
const DEFAULT_PASSWORD = 'Pa55w0rd';
const SALT_ROUNDS = 10;

async function init() {
  const hash = await bcrypt.hash(DEFAULT_PASSWORD, SALT_ROUNDS);
  await fs.writeFile(HASH_FILE, hash);
  console.log('Password hash file created successfully');
}

init().catch(console.error); 