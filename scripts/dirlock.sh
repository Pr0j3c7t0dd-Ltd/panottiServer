#!/bin/bash

LOCK_STATE_FILE="$HOME/.dir_locks"

usage() {
    echo "Usage:"
    echo "  $0 lock <directory>   - Lock a directory"
    echo "  $0 unlock <directory> - Unlock a directory"
    echo "  $0 list              - List locked directories"
}

check_directory() {
    if [ ! -d "$1" ]; then
        echo "Error: Directory '$1' does not exist"
        exit 1
    fi
}

get_absolute_path() {
    echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

store_permissions() {
    local dir="$1"
    local temp_file=$(mktemp)
    
    # Store permissions for directory and all contents
    find "$dir" -exec sh -c '
        for path; do
            if [ -e "$path" ]; then
                # Get permissions in octal format
                perms=$(stat -f "%OLp" "$path")
                echo "$path:$perms"
            fi
        done
    ' sh {} + > "$temp_file"
    
    echo "$temp_file"
}

lock_directory() {
    local dir=$(get_absolute_path "$1")
    check_directory "$dir"
    
    if grep -q "^$dir:" "$LOCK_STATE_FILE" 2>/dev/null; then
        echo "Directory '$dir' is already locked"
        exit 1
    fi
    
    echo "Saving original permissions..."
    local perm_file=$(store_permissions "$dir")
    
    echo "Applying lock..."
    # Remove write permissions recursively and add ACL
    if chmod -R a-w "$dir" 2>/dev/null && 
       sudo chmod -R +a "everyone deny write,delete,append,writeattr,writeextattr,chown" "$dir" 2>/dev/null; then
        # Store directory path and permissions
        echo "$dir:PERMS:$(cat "$perm_file")" >> "$LOCK_STATE_FILE"
        rm "$perm_file"
        echo "Directory '$dir' has been locked"
    else
        rm "$perm_file"
        echo "Error: Failed to lock directory '$dir'"
        exit 1
    fi
}

unlock_directory() {
    local dir=$(get_absolute_path "$1")
    check_directory "$dir"
    
    # Get all locked paths in this directory
    local paths_to_unlock=$(grep "^$dir" "$LOCK_STATE_FILE" 2>/dev/null)
    if [ -z "$paths_to_unlock" ]; then
        paths_to_unlock=$(grep "^$dir/.*" "$LOCK_STATE_FILE" 2>/dev/null)
        if [ -z "$paths_to_unlock" ]; then
            echo "Directory '$dir' is not locked by this script"
            exit 1
        fi
    fi
    
    echo "Removing access restrictions..."
    # Remove ACLs recursively from all files and directories
    find "$dir" -exec sudo chmod -RN {} \; 2>/dev/null
    
    # Restore original permissions for each locked path
    echo "Restoring original permissions..."
    echo "$paths_to_unlock" | while read -r line; do
        local current_path=$(echo "$line" | cut -d: -f1)
        if [ -e "$current_path" ]; then
            # Extract and apply permissions
            echo "$line" | sed 's/^.*:PERMS://' | while IFS=: read -r file mode; do
                if [ -e "$file" ] && [[ "$file" == "$current_path"* ]]; then
                    chmod "$mode" "$file"
                    echo "  Restored permissions for: $file"
                fi
            done
        fi
    done
    
    # Remove entries from lock state file
    # First, remove the directory itself
    sed -i '' "\#^$dir:#d" "$LOCK_STATE_FILE"
    # Then remove all subdirectories and files
    sed -i '' "\#^$dir/.*#d" "$LOCK_STATE_FILE"
    
    echo "Directory '$dir' and all contents have been unlocked"
}

list_locked() {
    if [ -f "$LOCK_STATE_FILE" ]; then
        echo "Currently locked directories:"
        echo "------------------------"
        while IFS= read -r line; do
            if [[ $line == /* ]]; then  # Only process lines starting with /
                dir=$(echo "$line" | cut -d: -f1)
                if [ -e "$dir" ]; then  # Check if path exists (file or directory)
                    if [ -d "$dir" ]; then
                        echo "📁 $dir"
                        echo "   Status: 🔒 LOCKED"
                        ls -ld "$dir"
                    else
                        echo "📄 $dir"
                        echo "   Status: 🔒 LOCKED"
                        ls -l "$dir"
                    fi
                    echo "------------------------"
                fi
            fi
        done < "$LOCK_STATE_FILE"
    else
        echo "No directories are currently locked"
    fi
}

case "$1" in
    "lock")
        if [ -z "$2" ]; then
            usage
            exit 1
        fi
        lock_directory "$2"
        ;;
    "unlock")
        if [ -z "$2" ]; then
            usage
            exit 1
        fi
        unlock_directory "$2"
        ;;
    "list")
        list_locked
        ;;
    *)
        usage
        exit 1
        ;;
esac