import asyncio, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select, func, or_, text
from app.database import async_session, init_db
from app.models import Drug

async def debug():
    await init_db()
    async with async_session() as session:
        # Test 1: Simple select
        result = await session.execute(select(Drug.name).limit(5))
        names = [r[0] for r in result.fetchall()]
        print("NAMES:", names)

        # Test 2: Raw SQL to test UPPER + LIKE in SQLite
        raw = await session.execute(text("SELECT name FROM drugs WHERE UPPER(name) LIKE '%WARFARIN%' LIMIT 5"))
        raw_names = [r[0] for r in raw.fetchall()]
        print("RAW SQL:", raw_names)

        # Test 3: The exact ORM query the endpoint uses
        query = "WARFARIN"
        stmt = select(Drug).where(
            or_(
                func.upper(Drug.name).contains(query),
                func.upper(Drug.generic_name).contains(query),
                func.upper(Drug.brand_names).contains(query)
            )
        ).limit(10)
        result2 = await session.execute(stmt)
        drugs = result2.scalars().all()
        print("ORM SEARCH:", [d.name for d in drugs])

asyncio.run(debug())
