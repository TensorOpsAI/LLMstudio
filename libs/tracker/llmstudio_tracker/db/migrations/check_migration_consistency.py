import os
import subprocess
import sys
from pathlib import Path

ALEMBIC_INI = "alembic.ini"

# Parse alembic.ini for script_location
def get_script_location(alembic_ini: str) -> Path:
    with open(alembic_ini) as f:
        for line in f:
            if line.strip().startswith("script_location"):
                _, value = line.split("=", 1)
                return Path(value.strip()) / "versions"
    raise RuntimeError("script_location not found in alembic.ini")


MIGRATIONS_DIR = get_script_location(ALEMBIC_INI)

print(f"🔍 Checking for missing Alembic migrations in {MIGRATIONS_DIR}...")

# Run alembic autogenerate
try:
    subprocess.run(
        [
            "poetry",
            "run",
            "alembic",
            "-c",
            ALEMBIC_INI,
            "revision",
            "--autogenerate",
            "-m",
            "check for changes",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
except subprocess.CalledProcessError as e:
    print("❌ Alembic revision failed")
    print(e.stderr.decode())
    sys.exit(1)

# Check if any files were modified
git_diff = subprocess.run(["git", "diff", "--exit-code"], stdout=subprocess.PIPE)
if git_diff.returncode != 0:
    print("❌ Detected uncommitted schema changes!")
    print("👉 Run: alembic revision --autogenerate -m 'your message'")
    print(f"👉 Then commit the migration file in {MIGRATIONS_DIR}")
    sys.exit(1)

# Get the most recent migration file
migration_files = sorted(
    MIGRATIONS_DIR.glob("*.py"), key=os.path.getmtime, reverse=True
)
if not migration_files:
    print("✅ No migration files found. All good.")
    sys.exit(0)

last_migration = migration_files[0]

# Clean up the temp migration if it has the expected message
if last_migration.read_text().startswith('"""check for changes'):
    print(f"🧹 Cleaning up temporary migration: {last_migration.name}")
    subprocess.run(["git", "restore", str(last_migration)], stderr=subprocess.DEVNULL)
    last_migration.unlink()

print("✅ Alembic migration check passed. Schema and migrations are in sync.")
