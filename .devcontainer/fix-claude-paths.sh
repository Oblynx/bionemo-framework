#!/usr/bin/env bash
# Build ~/.claude from the readonly-mounted ~/.claude_host:
#   - Symlinks for most files (transparent r/w back to host)
#   - Copies for files that need path rewriting (e.g. known_marketplaces.json)
# Runs as postStartCommand on every container start.
set -euo pipefail

host_dir="/home/ubuntu/.claude_host"
claude_dir="/home/ubuntu/.claude"
container_home="/home/ubuntu"

if [[ ! -d "$host_dir" ]]; then
    echo "fix-claude-paths: no host .claude mount found, skipping"
    exit 0
fi

# --- Build symlink tree ---
# Mirror the host directory structure under ~/.claude using symlinks.
# For each top-level entry in .claude_host, create a symlink unless
# it's in the "copy" list below.
mkdir -p "$claude_dir"

# Files/dirs that need to be COPIED (not symlinked) for path rewriting.
# Use paths relative to .claude_host.
copy_list=(
    "plugins/known_marketplaces.json"
)

# Helper: check if a relative path is in the copy list
is_copy_target() {
    local rel="$1"
    for item in "${copy_list[@]}"; do
        [[ "$rel" == "$item" ]] && return 0
    done
    return 1
}

# Symlink all top-level entries, skipping those that need special handling
for entry in "$host_dir"/*; do
    name=$(basename "$entry")
    target="${claude_dir}/${name}"

    # If this top-level dir contains copy targets, we need to handle it
    # specially: create real dir with symlinks inside, except for copy targets.
    has_copy_children=false
    for item in "${copy_list[@]}"; do
        if [[ "$item" == "$name/"* ]]; then
            has_copy_children=true
            break
        fi
    done

    if $has_copy_children; then
        # Create real directory, symlink children, copy exceptions
        mkdir -p "$target"
        for child in "$entry"/*; do
            child_name=$(basename "$child")
            rel_path="${name}/${child_name}"
            child_target="${target}/${child_name}"

            # Remove stale symlink/file
            rm -f "$child_target"

            if is_copy_target "$rel_path"; then
                cp "$child" "$child_target"
            else
                ln -sf "$child" "$child_target"
            fi
        done
    else
        # Simple top-level symlink — skip if target is already a real directory
        # (Claude Code creates dirs like backups/, cache/, sessions/ at runtime)
        if [[ -d "$target" && ! -L "$target" ]]; then
            continue
        fi
        rm -f "$target"
        ln -sf "$entry" "$target"
    fi
done
