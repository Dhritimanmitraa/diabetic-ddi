import asyncio
import sys
import os
from sqlalchemy import select, func

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.database import async_session
from app.models import Drug, DrugInteraction
import app.diabetic.models
import app.prescription.models

async def check():
    async with async_session() as session:
        # Check drugs count
        res = await session.execute(select(func.count(Drug.id)))
        drugs_count = res.scalar()
        
        # Check interactions count
        res = await session.execute(select(func.count(DrugInteraction.id)))
        interactions_count = res.scalar()
        
        print(f"Current Drugs count: {drugs_count}")
        print(f"Current Interactions count: {interactions_count}")

if __name__ == "__main__":
    asyncio.run(check())
