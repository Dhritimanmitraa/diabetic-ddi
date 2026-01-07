"""
Data service for querying TWOSIDES database
"""

import logging
import os
from typing import List, Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from .schemas import InteractionContext

logger = logging.getLogger(__name__)


class DataService:
    """Service for querying drug interaction database"""

    def __init__(self):
        self.db_path: Optional[str] = None
        self.engine = None
        self.async_session: Optional[async_sessionmaker] = None
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        """Check if database is connected"""
        return self._is_connected

    async def initialize(self) -> bool:
        """Initialize database connection"""
        try:
            # Try to find the database in the parent project
            possible_paths = [
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "..",
                    "backend",
                    "drug_interactions.db",
                ),
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "..",
                    "..",
                    "backend",
                    "drug_interactions.db",
                ),
                "C:\\Drug\\backend\\drug_interactions.db",
            ]

            for db_path in possible_paths:
                if os.path.exists(db_path):
                    self.db_path = db_path
                    break

            if not self.db_path or not os.path.exists(self.db_path):
                logger.warning(
                    "Database not found. TWOSIDES context will not be available."
                )
                return False

            # Create async engine for SQLite
            db_url = f"sqlite+aiosqlite:///{self.db_path}"
            self.engine = create_async_engine(db_url, echo=False)
            self.async_session = async_sessionmaker(
                self.engine, class_=AsyncSession, expire_on_commit=False
            )

            self._is_connected = True
            logger.info(f"Database connected: {self.db_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self._is_connected = False
            return False

    async def get_interaction_context(
        self, drug1: str, drug2: str
    ) -> Optional[InteractionContext]:
        """
        Get TWOSIDES interaction context for a drug pair
        """
        if not self._is_connected or not self.async_session:
            return None

        try:
            async with self.async_session() as session:
                # Query twosides_interactions table
                query = text(
                    """
                    SELECT effect, severity
                    FROM twosides_interactions
                    WHERE (
                        (LOWER(drug1_name) = LOWER(:drug1) AND LOWER(drug2_name) = LOWER(:drug2))
                        OR
                        (LOWER(drug1_name) = LOWER(:drug2) AND LOWER(drug2_name) = LOWER(:drug1))
                    )
                    LIMIT 50
                """
                )

                result = await session.execute(query, {"drug1": drug1, "drug2": drug2})
                rows = result.fetchall()

                if not rows:
                    return InteractionContext(
                        known_interaction=False, side_effects=[], interaction_count=0
                    )

                # Extract side effects
                side_effects = []
                for row in rows:
                    if row[0]:  # effect column
                        side_effects.append(str(row[0]))

                # Remove duplicates
                side_effects = list(set(side_effects))

                return InteractionContext(
                    known_interaction=True,
                    side_effects=side_effects[:20],  # Limit to 20
                    interaction_count=len(rows),
                )

        except Exception as e:
            logger.error(f"Error querying interaction context: {e}")
            return None

    async def search_drugs(self, query: str, limit: int = 10) -> List[str]:
        """
        Search for drugs by name
        """
        if not self._is_connected or not self.async_session:
            return []

        try:
            async with self.async_session() as session:
                search_query = text(
                    """
                    SELECT DISTINCT name
                    FROM drugs
                    WHERE LOWER(name) LIKE LOWER(:query)
                    ORDER BY name
                    LIMIT :limit
                """
                )

                result = await session.execute(
                    search_query, {"query": f"%{query}%", "limit": limit}
                )
                rows = result.fetchall()

                return [row[0] for row in rows if row[0]]

        except Exception as e:
            logger.error(f"Error searching drugs: {e}")
            return []

    async def get_drug_count(self) -> int:
        """Get total number of drugs in database"""
        if not self._is_connected or not self.async_session:
            return 0

        try:
            async with self.async_session() as session:
                query = text("SELECT COUNT(*) FROM drugs")
                result = await session.execute(query)
                count = result.scalar()
                return int(count) if count else 0
        except Exception as e:
            logger.error(f"Error getting drug count: {e}")
            return 0
