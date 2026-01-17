from alembic import op
import sqlalchemy as sa

revision = "0001_init"
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        "inference_logs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("endpoint", sa.String(length=50), nullable=False),
        sa.Column("request_json", sa.JSON(), nullable=False),
        sa.Column("response_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
    )
    op.create_index("ix_inference_logs_endpoint", "inference_logs", ["endpoint"])

def downgrade():
    op.drop_index("ix_inference_logs_endpoint", table_name="inference_logs")
    op.drop_table("inference_logs")
