import os
from logging.config import fileConfig

import llmstudio_tracker.base
from alembic import context
from dotenv import load_dotenv
from llmstudio_tracker.base_class import Base
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine.url import make_url

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


load_dotenv()

db_url = os.getenv("LLMSTUDIO_TRACKING_URI")
print(f"DB URL: {db_url}")
if not db_url:
    raise RuntimeError("LLMSTUDIO_TRACKING_URI not set in environment")

url_obj = make_url(db_url)
is_sqlite = url_obj.drivername.startswith("sqlite")


config.set_main_option("sqlalchemy.url", db_url)

alembic_table_name = (
    os.getenv("LLMSTUDIO_ALEMBIC_TABLE_NAME") or "llmstudio_alembic_version"
)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    context.configure(
        url=db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table=alembic_table_name,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    try:
        connectable = engine_from_config(
            config.get_section(config.config_ini_section, {}),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

        with connectable.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                render_as_batch=is_sqlite,
                compare_type=True,
                version_table=alembic_table_name,
            )

            with context.begin_transaction():
                context.run_migrations()
    except Exception as e:
        print(f"Error during migration: {e}")


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
