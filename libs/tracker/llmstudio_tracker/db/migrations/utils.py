import subprocess


def run_alembic_upgrade():
    try:
        if _alembic_is_up_to_date():
            print("Alembic: DB is already up-to-date.")
            return

        print("Alembic: DB is not up-to-date. Running Alembic upgrade...")
        subprocess.run(["poetry", "run", "alembic", "upgrade", "head"], check=True)
        print("Alembic: Upgrade successful.")
    except subprocess.CalledProcessError as e:
        print(f"Alembic: Upgrade failed: {e}")
        raise


def _alembic_is_up_to_date() -> bool:
    """Returns True if the DB is already at the latest revision."""
    try:
        current = (
            subprocess.check_output(
                ["poetry", "run", "alembic", "current"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )

        head = (
            subprocess.check_output(
                ["poetry", "run", "alembic", "heads"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )

        current_rev_line = next(
            (
                line
                for line in current.splitlines()
                if line.strip() and not line.startswith("DB URL")
            ),
            "",
        )
        head_rev_line = next(
            (
                line
                for line in head.splitlines()
                if line.strip() and not line.startswith("DB URL")
            ),
            "",
        )

        current_rev = current_rev_line.split(" ")[0]
        head_rev = head_rev_line.split(" ")[0]

        return current_rev == head_rev

    except subprocess.CalledProcessError as e:
        print("Alembic: Check if up-to-date failed. Upgrading head to be safe.")
        return False
