## Scripts Directory

### dirlock.sh

A utility script for managing directory locks using Linux's immutable flag (`chattr`).

#### Features

- Lock directories recursively to prevent modifications
- Unlock previously locked directories
- List all currently locked directories
- Tracks locked directories in `~/.dir_locks` for easy management

#### Prerequisites

- Linux system with `chattr` command available
- Sudo privileges (required for setting/removing immutable flags)

#### Installation

1. Ensure the script is executable:
```bash
chmod +x dirlock.sh
```

#### Usage

```bash
./dirlock.sh lock /path/to/directory    # Lock a directory
./dirlock.sh unlock /path/to/directory  # Unlock a directory
./dirlock.sh list                       # See what's locked
```

#### Examples

```bash
# Lock a directory
./dirlock.sh lock ~/important_data

# View locked directories
./dirlock.sh list

# Unlock a directory
./dirlock.sh unlock ~/important_data
```

#### Notes

- The script uses sudo for the `chattr` commands since they require root privileges
- Locked directories are tracked in `~/.dir_locks`
- The script will verify directory existence before attempting operations
- Absolute paths are used internally to avoid ambiguity 