#!/usr/bin/env bash
# Updates lines that start with "local HOME" inside .do files so they point to
# the directory of each .do file (absolute path). Works in bash or zsh.

set -euo pipefail
# make zsh behave like POSIX sh where it matters
if [ -n "${ZSH_VERSION:-}" ]; then
  emulate -L sh
  set -o pipefail
fi

# Find all .do files (recursively), robust to spaces/newlines in paths
find . -type f -name '*.do' -print0 |
while IFS= read -r -d '' file; do
  # Absolute, physical path of the .do file's directory
  file_dir="$(cd "$(dirname "$file")" && pwd -P)"

  tmp="$(mktemp)"
  # Replace any line that *starts with* optional spaces + "local HOME"
  # with:  local HOME "<absolute dir>"
  awk -v rep="local HOME \"${file_dir}\"" '
    /^[[:space:]]*local[[:space:]]+HOME([[:space:]]|$)/ { print rep; next }
    { print }
  ' "$file" > "$tmp"

  # Only overwrite if something actually changed
  if ! cmp -s "$tmp" "$file"; then
    mv "$tmp" "$file"
    printf 'Updated: %s -> %s\n' "$file" "$file_dir"
  else
    rm -f "$tmp"
  fi
done
