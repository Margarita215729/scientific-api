import os
import asyncpg
import sqlite3
from typing import Optional, Dict, Any, List
import json
from datetime import datetime, timedelta

class DatabaseConfig:
    """Database configuration and connection manager"""
    
    def __init__(self):
        self.db_type = os.getenv("DB_TYPE", "sqlite")  # sqlite or postgresql
        self.db_url = os.getenv("DATABASE_URL", "sqlite:///scientific_api.db")
        self.connection = None
        
    async def connect(self):
        """Connect to database"""
        if self.db_type == "postgresql":
            self.connection = await asyncpg.connect(self.db_url)
        else:
            # SQLite for local development
            self.connection = sqlite3.connect("scientific_api.db")
            self.connection.row_factory = sqlite3.Row
            
    async def disconnect(self):
        """Disconnect from database"""
        if self.connection:
            if self.db_type == "postgresql":
                await self.connection.close()
            else:
                self.connection.close()
                
    async def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute a query and return results"""
        if not self.connection:
            await self.connect()
            
        try:
            if self.db_type == "postgresql":
                if params:
                    result = await self.connection.fetch(query, *params)
                else:
                    result = await self.connection.fetch(query)
                return [dict(row) for row in result]
            else:
                # SQLite synchronous operations
                cursor = self.connection.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                if query.strip().upper().startswith("SELECT"):
                    result = cursor.fetchall()
                    return [dict(row) for row in result]
                else:
                    self.connection.commit()
                    return []
        except Exception as e:
            print(f"Database error: {e}")
            return []
            
    async def init_database(self):
        """Initialize database with schema"""
        schema_file = "database/schema.sql"
        if os.path.exists(schema_file):
            with open(schema_file, 'r') as f:
                schema = f.read()
                
            # Split schema into individual statements
            statements = [stmt.strip() for stmt in schema.split(';') if stmt.strip()]
            
            for statement in statements:
                if statement:
                    await self.execute_query(statement)
                    
    async def get_astronomical_objects(self, limit: int = 100, object_type: str = None) -> List[Dict]:
        """Get astronomical objects from database"""
        query = "SELECT * FROM astronomical_objects"
        params = []
        
        if object_type:
            query += " WHERE object_type = ?"
            params.append(object_type)
            
        query += f" LIMIT {limit}"
        
        return await self.execute_query(query, tuple(params) if params else None)
        
    async def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats_query = "SELECT metric_name, metric_value, metric_unit FROM system_statistics"
        stats_result = await self.execute_query(stats_query)
        
        statistics = {}
        for row in stats_result:
            statistics[row['metric_name']] = {
                'value': row['metric_value'],
                'unit': row['metric_unit']
            }
            
        return statistics
        
    async def cache_api_response(self, cache_key: str, data: Dict, expires_in_hours: int = 24):
        """Cache API response"""
        expires_at = datetime.now() + timedelta(hours=expires_in_hours)
        
        query = """
        INSERT OR REPLACE INTO api_cache (cache_key, cache_value, expires_at)
        VALUES (?, ?, ?)
        """
        
        await self.execute_query(query, (cache_key, json.dumps(data), expires_at))
        
    async def get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Get cached API response"""
        query = """
        SELECT cache_value FROM api_cache 
        WHERE cache_key = ? AND expires_at > ?
        """
        
        result = await self.execute_query(query, (cache_key, datetime.now()))
        
        if result:
            return json.loads(result[0]['cache_value'])
        return None
        
    async def log_search(self, user_id: int, search_type: str, query: str, 
                        params: Dict, results_count: int, execution_time: int):
        """Log search history"""
        insert_query = """
        INSERT INTO search_history 
        (user_id, search_type, search_query, search_params, results_count, execution_time_ms)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        await self.execute_query(insert_query, (
            user_id, search_type, query, json.dumps(params), 
            results_count, execution_time
        ))

# Global database instance
db = DatabaseConfig() 