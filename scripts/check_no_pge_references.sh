#!/usr/bin/env bash
# Fail if PG&E references appear outside agent-only docs (gitignored locally).
set -euo pipefail

PATTERN='PG&E|Pacific Gas|PG&E Geospatial'
PATHS=(src tests config pyproject.toml README.md main.py)

fail() {
  echo "ERROR: PG&E references found in public/shipped files (see matches above)."
  exit 1
}

if command -v rg >/dev/null 2>&1; then
  if rg -i "$PATTERN" \
    --glob '!AGENT.md' \
    --glob '!CLAUDE.md' \
    --glob '!PROJECT_SPEC.md' \
    "${PATHS[@]}"; then
    fail
  fi
elif grep -riE \
  --exclude-dir=__pycache__ \
  --exclude='*.pyc' \
  "$PATTERN" "${PATHS[@]}" >/dev/null 2>&1; then
  grep -riE --color=never \
    --exclude-dir=__pycache__ \
    --exclude='*.pyc' \
    "$PATTERN" "${PATHS[@]}" || true
  fail
fi
