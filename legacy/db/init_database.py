#!/usr/bin/env python3
"""
Database initialization script for Scientific API
"""
import asyncio
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.config import db

async def init_database():
    """Initialize the database with schema and sample data"""
    try:
        print("🔄 Initializing database...")
        await db.init_database()
        print("✅ Database schema created successfully")
        
        # Test database connection
        stats = await db.get_statistics()
        print(f"📊 Database statistics: {len(stats)} metrics loaded")
        
        objects = await db.get_astronomical_objects(limit=5)
        print(f"🌌 Sample objects: {len(objects)} astronomical objects loaded")
        
        print("🎉 Database initialization completed successfully!")
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        sys.exit(1)
    finally:
        await db.disconnect()

if __name__ == "__main__":
    asyncio.run(init_database()) 