"""initial_schema

Revision ID: 3e5cdfa4a48e
Revises: 
Create Date: 2026-03-09 02:00:27.115768

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3e5cdfa4a48e'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Initial schema baseline — tables already exist via create_all()."""
    pass


def downgrade() -> None:
    """Nothing to revert for the baseline."""
    pass
