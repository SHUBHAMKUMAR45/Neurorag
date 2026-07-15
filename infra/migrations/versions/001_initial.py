"""Initial schema: queries, eval_metrics, query_memory tables

Revision ID: 001_initial
Revises:
Create Date: 2026-04-13 00:00:00
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── queries ──────────────────────────────────────────────────────────────
    op.create_table(
        "queries",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("query", sa.Text, nullable=False),
        sa.Column("answer", sa.Text),
        sa.Column("citations", postgresql.ARRAY(sa.Text)),
        sa.Column("confidence", sa.Float),
        sa.Column("loops", sa.Integer),
        sa.Column("latency_ms", sa.Integer),
        sa.Column("insufficient", sa.Boolean, server_default="false"),
        sa.Column("from_cache", sa.Boolean, server_default="false"),
        sa.Column("hint_applied", sa.Boolean, server_default="false"),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("idx_queries_created_at", "queries", ["created_at"], postgresql_using="brin")
    op.create_index("idx_queries_confidence", "queries", ["confidence"])

    # ── eval_metrics ─────────────────────────────────────────────────────────
    op.create_table(
        "eval_metrics",
        sa.Column("id", sa.Text, sa.ForeignKey("queries.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("faithfulness", sa.Float),
        sa.Column("relevance", sa.Float),
        sa.Column("completeness", sa.Float),
        sa.Column("failure_types", postgresql.ARRAY(sa.Text)),
        sa.Column("hallucination", sa.Boolean, server_default="false"),
        sa.Column("evaluated_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("idx_eval_faithfulness", "eval_metrics", ["faithfulness"])
    op.create_index("idx_eval_hallucination", "eval_metrics", ["hallucination"])

    # ── query_memory ──────────────────────────────────────────────────────────
    op.create_table(
        "query_memory",
        sa.Column("query_hash", sa.Text, primary_key=True),
        sa.Column("query", sa.Text, nullable=False),
        sa.Column("answer", sa.Text),
        sa.Column("citations", postgresql.ARRAY(sa.Text)),
        sa.Column("confidence", sa.Float),
        sa.Column("loops", sa.Integer),
        sa.Column("latency_ms", sa.Integer),
        sa.Column("failure_types", postgresql.ARRAY(sa.Text)),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("idx_memory_confidence", "query_memory", ["confidence"])
    op.create_index("idx_memory_updated_at", "query_memory", ["updated_at"])


def downgrade() -> None:
    op.drop_table("query_memory")
    op.drop_table("eval_metrics")
    op.drop_table("queries")
