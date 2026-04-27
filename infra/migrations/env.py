"""
Alembic migration environment configuration for NeuroRAG.
Compatible with FastAPI async runtime + Alembic sync migrations.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool
from dotenv import load_dotenv

# ---------------------------------------------------
# Load .env file
# ---------------------------------------------------
load_dotenv()

# ---------------------------------------------------
# Alembic Config
# ---------------------------------------------------
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---------------------------------------------------
# Read DB URL from environment
# App may use asyncpg, but Alembic needs psycopg2
# ---------------------------------------------------
db_url = os.environ.get(
    "POSTGRES_URL",
    "postgresql+asyncpg://neurorag_user:StrongPass123@localhost:5432/neurorag"
)

# Convert async driver -> sync driver for Alembic
if db_url.startswith("postgresql+asyncpg://"):
    db_url = db_url.replace(
        "postgresql+asyncpg://",
        "postgresql+psycopg2://",
        1
    )

# Apply URL to Alembic
config.set_main_option("sqlalchemy.url", db_url)

# ---------------------------------------------------
# Metadata (set models metadata here later if needed)
# Example:
# from app.models import Base
# target_metadata = Base.metadata
# ---------------------------------------------------
target_metadata = None


# ---------------------------------------------------
# Offline migrations
# ---------------------------------------------------
def run_migrations_offline() -> None:
    """Run migrations in offline mode."""
    
    url = config.get_main_option("sqlalchemy.url")

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------
# Online migrations
# ---------------------------------------------------
def run_migrations_online() -> None:
    """Run migrations in online mode."""

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        future=True,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


# ---------------------------------------------------
# Entry
# ---------------------------------------------------
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()