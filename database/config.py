import os
import asyncpg
import sqlite3
from typing import Optional, Dict, Any, List
import json
from datetime import datetime, timedelta

from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos.exceptions import CosmosResourceNotFoundError, CosmosHttpResponseError

class DatabaseConfig:
    """Database configuration and connection manager"""
    
    def __init__(self):
        self.db_type = os.getenv("DB_TYPE", "sqlite")  # sqlite, postgresql, or cosmosdb
        self.db_url = os.getenv("DATABASE_URL", "sqlite:///scientific_api.db") # For SQL
        
        self.cosmos_endpoint = os.getenv("COSMOS_ENDPOINT")
        self.cosmos_key = os.getenv("COSMOS_KEY")
        self.cosmos_database_name = os.getenv("COSMOS_DATABASE_NAME", "ScientificDB")
        
        self.connection = None # For SQL
        self.cosmos_client = None
        self.database_client = None
        
        self.container_clients: Dict[str, Any] = {}

        # Define Cosmos DB containers and their partition keys
        # Adjust partition keys based on your query patterns for better performance/cost
        self.cosmos_containers_config = {
            "users": "/username", # Assuming username is unique and often used in queries
            "api_keys": "/user_id",
            "astronomical_objects": "/object_id", # From schema.sql, object_id is unique
            "astronomical_data": "/object_id",   # Assuming data is tied to an astronomical_object
            "search_history": "/user_id",        # If searches are often queried per user
            "ml_analysis_results": "/user_id",   # If results are often queried per user
            "api_cache": "/cache_key",           # cache_key is unique
            "system_statistics": "/metric_name"  # Or a static key like "/singleton" if data is small
        }
        
    async def connect(self):
        """Connect to the configured database type"""
        if self.db_type == "cosmosdb":
            if not self.cosmos_endpoint or not self.cosmos_key:
                raise ValueError("COSMOS_ENDPOINT and COSMOS_KEY must be set for Cosmos DB")
            print(f"[DB CONNECT {self.db_type.upper()}] Connecting to {self.cosmos_endpoint}...")
            self.cosmos_client = CosmosClient(self.cosmos_endpoint, credential=self.cosmos_key)
            self.database_client = self.cosmos_client.get_database_client(self.cosmos_database_name)
            print(f"[DB CONNECT {self.db_type.upper()}] Connected. Database client for '{self.cosmos_database_name}' obtained.")
            # Pre-initialize container clients for defined configs
            for container_name in self.cosmos_containers_config.keys():
                self._get_container_client(container_name)

        elif self.db_type == "postgresql":
            print(f"[DB CONNECT {self.db_type.upper()}] Connecting to {self.db_url}...")
            self.connection = await asyncpg.connect(self.db_url)
            print(f"[DB CONNECT {self.db_type.upper()}] Connected.")
        else: # Default to SQLite
            print(f"[DB CONNECT {self.db_type.upper()}] Connecting to scientific_api.db...")
            self.connection = sqlite3.connect("scientific_api.db")
            self.connection.row_factory = sqlite3.Row
            print(f"[DB CONNECT {self.db_type.upper()}] Connected.")
            
    async def disconnect(self):
        """Disconnect from database"""
        if self.cosmos_client:
            # CosmosClient doesn't have an explicit async close. Connections are managed by the underlying HTTP client.
            # Set to None to allow re-initialization if needed.
            self.cosmos_client = None
            self.database_client = None
            self.container_clients = {}
            print(f"[DB DISCONNECT {self.db_type.upper()}] Client resources released.")
        if self.connection:
            if self.db_type == "postgresql":
                await self.connection.close()
            else:
                self.connection.close()
            print(f"[DB DISCONNECT {self.db_type.upper()}] Connection closed.")
                
    def _get_container_client(self, container_name: str):
        if not self.database_client and self.db_type == "cosmosdb":
            # This should ideally not happen if connect() was called
            print(f"[DB WARN] database_client not initialized while getting container for {container_name}. Attempting to connect.")
            # This is tricky in async context from a sync method, best to ensure connect is always called first.
            # For simplicity, assume connect has been called.
            raise RuntimeError("Database not connected. Call connect() first.")

        if container_name not in self.container_clients and self.database_client:
            print(f"[DB COSMOS] Getting container client for '{container_name}'...")
            try:
                self.container_clients[container_name] = self.database_client.get_container_client(container_name)
                print(f"[DB COSMOS] Container client for '{container_name}' obtained.")
            except CosmosResourceNotFoundError:
                print(f"[DB COSMOS WARN] Container '{container_name}' not found. It might be created during init_database.")
                # We might allow this to proceed and let init_database handle creation
                # For operations, this would mean the container doesn't exist yet.
                return None # Or raise error, depending on desired behavior
        return self.container_clients.get(container_name)

    async def execute_query(self, query: str, params: tuple = None, container_name: Optional[str] = None) -> List[Dict]:
        """
        Execute a query. For Cosmos DB, this is a read query.
        For SQL, it can be any query.
        `container_name` is required for Cosmos DB queries.
        """
        if not self.connection and not self.cosmos_client:
            await self.connect() # Ensure connection is established
            
        try:
            if self.db_type == "cosmosdb":
                if not container_name:
                    raise ValueError("container_name is required for Cosmos DB queries")
                container_client = self._get_container_client(container_name)
                if not container_client:
                     print(f"[DB COSMOS ERROR] Container '{container_name}' does not exist or client not initialized.")
                     return []

                print(f"[DB COSMOS QUERY] Executing on '{container_name}': {query} with params {params}")
                
                # Cosmos DB uses named parameters like @paramName
                # This basic version assumes direct query string, advanced use would map params
                # For simplicity, if params are provided, assume they are in the correct format for the query string.
                # A more robust solution would inspect 'params' and build the 'parameters' list for query_items.
                
                # Simplified execution (may need adjustment based on how query/params are structured):
                # This example assumes `params` are directly usable or the query is simple.
                # The `enable_cross_partition_query` might be needed if partition key is not in query.
                
                # Let's assume 'params' are the values for SQL-style '?' placeholders
                # This is a rough adaptation and might not cover all SQL to Cosmos query translations
                if params:
                    # This is a placeholder for actual parameter handling.
                    # Direct substitution is generally unsafe and not how Cosmos SDK works.
                    # For now, we'll assume simple queries or that `query_items` handles it.
                    # A real implementation would map `params` to Cosmos DB's `parameters` argument.
                    print(f"[DB COSMOS WARN] Parameter handling in execute_query for Cosmos DB is simplified and may not work for all cases.")
                
                query_results = container_client.query_items(
                    query=query,
                    # parameters= [{"name": f"@p{i}", "value": v} for i, v in enumerate(params)] if params else None, # Example
                    enable_cross_partition_query=True # Be mindful of cost/performance
                )
                async for item in query_results:
                    items.append(item)
                return items

            elif self.db_type == "postgresql":
                if params:
                    result = await self.connection.fetch(query, *params)
                else:
                    result = await self.connection.fetch(query)
                return [dict(row) for row in result]
            else: # SQLite
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
        except CosmosHttpResponseError as e:
            print(f"[DB COSMOS ERROR] Cosmos DB HTTP error: {e.status_code}, {e.message}")
            return []
        except Exception as e:
            print(f"Database error ({self.db_type}): {e}")
            import traceback
            traceback.print_exc()
            return []

    async def upsert_item(self, container_name: str, item_body: Dict) -> Optional[Dict]:
        """Upserts (creates or replaces) an item in the specified Cosmos DB container."""
        if self.db_type != "cosmosdb":
            print(f"[DB {self.db_type.upper()} ERROR] upsert_item is only for Cosmos DB.")
            return None
        
        if not self.cosmos_client: # Ensure client is available
            await self.connect()

        container_client = self._get_container_client(container_name)
        if not container_client:
            print(f"[DB COSMOS ERROR] Container '{container_name}' not found for upsert.")
            return None
        try:
            print(f"[DB COSMOS UPSERT] Upserting item into '{container_name}': {str(item_body)[:100]}...")
            response = await container_client.upsert_item(body=item_body)
            print(f"[DB COSMOS UPSERT] Item upserted successfully into '{container_name}'.")
            return response
        except CosmosHttpResponseError as e:
            print(f"[DB COSMOS ERROR] Failed to upsert item into '{container_name}': {e}")
            return None
        except Exception as e:
            print(f"[DB GENERAL ERROR] Unexpected error during upsert into '{container_name}': {e}")
            return None

    async def init_database(self):
        """Initialize database: create database and containers for Cosmos DB, or run schema.sql for SQL DBs."""
        if not self.connection and not self.cosmos_client: # Ensure client/connection is available
             await self.connect()

        print(f"[DB INIT {self.db_type.upper()}] Starting database initialization...")
        
        if self.db_type == "cosmosdb":
            if not self.database_client:
                 print(f"[DB INIT COSMOS ERROR] Cosmos DB client not initialized.")
                 return

            try:
                print(f"[DB INIT COSMOS] Ensuring database '{self.cosmos_database_name}' exists...")
                await self.cosmos_client.create_database_if_not_exists(id=self.cosmos_database_name)
                print(f"[DB INIT COSMOS] Database '{self.cosmos_database_name}' ensured.")
                self.database_client = self.cosmos_client.get_database_client(self.cosmos_database_name) # Re-get after creation

                for container_name, partition_key_path in self.cosmos_containers_config.items():
                    print(f"[DB INIT COSMOS] Ensuring container '{container_name}' with partition key '{partition_key_path}' exists...")
                    try:
                        await self.database_client.create_container_if_not_exists(
                            id=container_name, 
                            partition_key=PartitionKey(path=partition_key_path)
                        )
                        print(f"[DB INIT COSMOS] Container '{container_name}' ensured.")
                        # Update container client cache
                        self.container_clients[container_name] = self.database_client.get_container_client(container_name)
                    except CosmosHttpResponseError as e:
                        print(f"[DB INIT COSMOS ERROR] Failed to create/ensure container '{container_name}': {e.message}")
                    except Exception as e_cont:
                        print(f"[DB INIT COSMOS ERROR] Unexpected error for container '{container_name}': {e_cont}")


                # Note: Seeding initial data (like default admin user or sample objects from schema.sql)
                # needs to be done here using self.upsert_item() for Cosmos DB.
                # This is different from executing schema.sql.
                # Example for users (assuming 'id' is generated by Cosmos or part of item_body):
                # await self.upsert_item("users", {"id": "admin", "username": "admin", "email": "admin@example.com", ...})
                # print("[DB INIT COSMOS] Initial data seeding would go here if implemented.")

            except CosmosHttpResponseError as e:
                print(f"[DB INIT COSMOS ERROR] Cosmos DB HTTP error during init: {e.status_code}, {e.message}")
            except Exception as e_main:
                print(f"[DB INIT COSMOS ERROR] Unexpected error during Cosmos DB init: {e_main}")

        elif os.path.exists("database/schema.sql"): # SQL databases
            print(f"[DB INIT {self.db_type.upper()}] Found schema file: database/schema.sql")
            with open("database/schema.sql", 'r') as f:
                schema = f.read()
            
            statements = [stmt.strip() for stmt in schema.split(';') if stmt.strip()]
            print(f"[DB INIT {self.db_type.upper()}] Found {len(statements)} statements to execute.")
            
            for i, statement in enumerate(statements):
                if statement:
                    print(f"[DB INIT {self.db_type.upper()}] Executing statement {i+1}/{len(statements)}: {statement[:100]}...")
                    try:
                        await self.execute_query(statement) # For SQL, execute_query handles DDL
                        print(f"[DB INIT {self.db_type.upper()}] Statement {i+1} executed successfully.")
                    except Exception as e_sql:
                        print(f"[DB INIT {self.db_type.upper()} ERROR] executing statement {i+1}: {statement[:100]}... - Error: {e_sql}")
            print(f"[DB INIT {self.db_type.upper()}] All SQL statements executed.")
        else:
            print(f"[DB INIT {self.db_type.upper()} ERROR] Schema file not found at database/schema.sql")
        print(f"[DB INIT {self.db_type.upper()}] Database initialization finished.")
            
    async def get_astronomical_objects(self, limit: int = 100, object_type: str = None) -> List[Dict]:
        """Get astronomical objects from database"""
        container_name = "astronomical_objects"
        
        if self.db_type == "cosmosdb":
            # Basic query, adjust for specific fields and filtering if needed
            query = f"SELECT * FROM c"
            if object_type:
                # This assumes there's a field object_type in your Cosmos DB items.
                # Parameters should be used for security and correctness.
                query += f" WHERE c.object_type = @object_type" 
            # TOP N for limit
            # A full query might be: "SELECT TOP @limit * FROM c WHERE c.object_type = @object_type"
            # For simplicity, applying limit after fetching, which is not optimal for large datasets.
            
            # Corrected query for Cosmos DB with TOP and potential filter
            query_parts = ["SELECT "]
            if limit > 0 :
                 query_parts.append(f"TOP {limit} ") # Cosmos DB uses TOP N
            query_parts.append("* FROM c")

            params_list = []
            conditions = []

            if object_type:
                conditions.append("c.object_type = @object_type_param")
                params_list.append({"name": "@object_type_param", "value": object_type})
            
            if conditions:
                query_parts.append(" WHERE ")
                query_parts.append(" AND ".join(conditions))
            
            final_query = "".join(query_parts)

            container_client = self._get_container_client(container_name)
            if not container_client: return []
            
            print(f"[DB COSMOS GET] Querying '{container_name}': {final_query} with params {params_list}")
            try:
                items = []
                async for item in container_client.query_items(
                    query=final_query, 
                    parameters=params_list if params_list else None, 
                    enable_cross_partition_query=True # Set to False if partition key is always part of query
                ):
                    items.append(item)
                # Cosmos DB TOP N handles limit, so no post-fetch slicing needed if used correctly
                return items[:limit] if limit > 0 and not "TOP" in final_query else items

            except CosmosHttpResponseError as e:
                print(f"[DB COSMOS ERROR] get_astronomical_objects: {e}")
                return []
        else: # SQL
            query = "SELECT * FROM astronomical_objects"
            sql_params_list = []
            if object_type:
                query += " WHERE object_type = ?"
                sql_params_list.append(object_type)
            if limit > 0:
                 query += f" LIMIT ?"
                 sql_params_list.append(limit)
            return await self.execute_query(query, tuple(sql_params_list) if sql_params_list else None)

    async def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        container_name = "system_statistics"
        if self.db_type == "cosmosdb":
            query = "SELECT * FROM c" # Get all stats documents
            container_client = self._get_container_client(container_name)
            if not container_client: return {}
            
            print(f"[DB COSMOS GET STATS] Querying '{container_name}': {query}")
            try:
                statistics = {}
                async for item in container_client.query_items(query=query, enable_cross_partition_query=True):
                    # Assuming item structure is like: {"id": "metric_name", "metric_name": "...", "value": ..., "unit": ...}
                    # or metric_name is a field within the doc. Adjust based on actual item structure.
                    # If 'id' is the metric_name, then item['id'] or item['metric_name']
                    metric_name = item.get('metric_name', item.get('id')) # Get metric name
                    if metric_name:
                        statistics[metric_name] = {
                            'value': item.get('metric_value', item.get('value')), # Be flexible with field names
                            'unit': item.get('metric_unit', item.get('unit'))
                        }
                return statistics
            except CosmosHttpResponseError as e:
                print(f"[DB COSMOS ERROR] get_statistics: {e}")
                return {}
        else: # SQL
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
        container_name = "api_cache"
        expires_at_dt = datetime.utcnow() + timedelta(hours=expires_in_hours) # Use UTC for consistency
        # Cosmos DB prefers ISO 8601 string for datetime
        expires_at_iso = expires_at_dt.isoformat() + "Z"


        if self.db_type == "cosmosdb":
            item_body = {
                "id": cache_key, # Use cache_key as the item ID for point reads
                "cache_key": cache_key, 
                "cache_value": data, # Store data directly, Cosmos handles JSON
                "expires_at": expires_at_iso,
                "_ts": int(datetime.utcnow().timestamp()) # Optional: manual timestamp if needed
            }
            await self.upsert_item(container_name, item_body)
        else: # SQL
            # SQLite stores datetime as text or real, ensure consistency
            expires_at_sql = expires_at_dt if self.db_type == "postgresql" else expires_at_iso
            query = """
            INSERT OR REPLACE INTO api_cache (cache_key, cache_value, expires_at)
            VALUES (?, ?, ?)
            """
            await self.execute_query(query, (cache_key, json.dumps(data), expires_at_sql)) #json.dumps for SQL
        
    async def get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Get cached API response"""
        container_name = "api_cache"
        current_time_iso = datetime.utcnow().isoformat() + "Z"

        if self.db_type == "cosmosdb":
            container_client = self._get_container_client(container_name)
            if not container_client: return None
            try:
                # Point read using ID is most efficient if cache_key is the ID
                print(f"[DB COSMOS CACHE GET] Reading item '{cache_key}' from '{container_name}'")
                item = await container_client.read_item(item=cache_key, partition_key=cache_key) # Assumes cache_key is also partition key
                
                if item and item.get('expires_at') > current_time_iso:
                    return item.get('cache_value')
                return None
            except CosmosResourceNotFoundError:
                print(f"[DB COSMOS CACHE GET] Cache miss for key '{cache_key}' (not found).")
                return None
            except CosmosHttpResponseError as e:
                print(f"[DB COSMOS CACHE GET ERROR] for key '{cache_key}': {e}")
                return None
        else: # SQL
            # For SQLite, ensure datetime comparison works. Storing as ISO string helps.
            current_time_sql = datetime.utcnow() if self.db_type == "postgresql" else current_time_iso
            query = """
            SELECT cache_value FROM api_cache 
            WHERE cache_key = ? AND expires_at > ?
            """
            result = await self.execute_query(query, (cache_key, current_time_sql))
            if result:
                return json.loads(result[0]['cache_value'])
            return None
        
    async def log_search(self, user_id: str, search_type: str, query: str, 
                        params: Dict, results_count: int, execution_time: int):
        """Log search history. For Cosmos DB, user_id could be part of a composite ID or a property."""
        container_name = "search_history"
        if self.db_type == "cosmosdb":
            item_id = f"{user_id}_{datetime.utcnow().timestamp()}" # Example unique ID
            item_body = {
                "id": item_id,
                "user_id": user_id,
                "search_type": search_type,
                "search_query": query,
                "search_params": params, # Stored as a nested object
                "results_count": results_count,
                "execution_time_ms": execution_time,
                "created_at": datetime.utcnow().isoformat() + "Z"
            }
            await self.upsert_item(container_name, item_body)
        else: # SQL
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