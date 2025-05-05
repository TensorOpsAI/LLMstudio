#!/bin/bash
set -e

# You can optionally pass a custom alembic.ini path
ALEMBIC_INI=${1:-alembic.ini}

# Get migration script location from alembic.ini
SCRIPT_LOCATION=$(grep -E "^script_location\s*=" "$ALEMBIC_INI" | sed 's/^script_location\s*=\s*//')
MIGRATIONS_DIR="${SCRIPT_LOCATION}/versions"

echo "🔍 Checking for missing Alembic migrations in $MIGRATIONS_DIR..."

# Generate a temporary migration
alembic -c "$ALEMBIC_INI" revision --autogenerate -m "check for changes"

# Check if any files changed (i.e. a new migration was generated)
if ! git diff --exit-code > /dev/null; then
  echo "❌ Detected uncommitted schema changes!"
  echo "👉 Run: alembic revision --autogenerate -m 'your message'"
  echo "👉 Then commit the migration file in $MIGRATIONS_DIR"
  exit 1
fi

# Get the last migration created
last_migration=$(ls -t "$MIGRATIONS_DIR" | head -n1)

# If it's our check file, clean it up
if grep -q "check for changes" "${MIGRATIONS_DIR}/${last_migration}"; then
  echo "🧹 Cleaning up temporary migration: $last_migration"
  git restore "${MIGRATIONS_DIR}/${last_migration}" 2>/dev/null || true
  rm -f "${MIGRATIONS_DIR}/${last_migration}"
fi

echo "✅ Alembic migration check passed. Schema and migrations are in sync."
