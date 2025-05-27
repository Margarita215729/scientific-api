import os
import asyncio
import sqlite3
import asyncpg
from typing import Optional, Dict, Any, List
import json
from datetime import datetime, timedelta
import logging

# MongoDB Async Driver
try:
    import motor.motor_asyncio
    MOTOR_AVAILABLE = True
except ImportError:
    motor = None
    MOTOR_AVAILABLE = False

logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Database configuration and connection manager for MongoDB (CosmosDB MongoDB API), PostgreSQL, and SQLite."""
    
    def __init__(self):
        self.db_type = os.getenv("DB_TYPE", "sqlite") 
        
        # MongoDB / CosmosDB (MongoDB API) settings
        self.mongodb_connection_string = os.getenv("AZURE_COSMOS_CONNECTION_STRING") # Используем эту переменную
        self.mongodb_database_name = os.getenv("COSMOS_DATABASE_NAME", "scientificdata") # Имя БД из строки подключения или по умолчанию
                                                                                       # В вашем случае это 'scientific-data'

        # SQL settings (PostgreSQL or SQLite)
        self.sql_db_url = os.getenv("DATABASE_URL", "sqlite:///./scientific_api.db")
        
        self.mongo_client = None      # motor client
        self.mongo_db = None          # motor database object
        self.sql_connection = None    # asyncpg or sqlite3 connection
        
        # Определяем имена коллекций (эквивалент таблиц)
        # Ключи партиционирования (shard keys) для CosmosDB MongoDB API настраиваются на уровне коллекции при её создании в Azure.
        # Здесь мы просто указываем имена коллекций.
        self.mongo_collections = [
            "users", "api_keys", "astronomical_objects", "astronomical_data",
            "search_history", "ml_analysis_results", "api_cache", "system_statistics"
        ]
        
        if self.db_type == "cosmosdb_mongo" and not MOTOR_AVAILABLE:
            logger.error("DB_TYPE is set to cosmosdb_mongo, but motor library is not available.")
            raise ImportError("motor library is required for MongoDB API interaction.")
        if self.db_type == "cosmosdb_mongo" and not self.mongodb_connection_string:
            logger.error("DB_TYPE is cosmosdb_mongo, but AZURE_COSMOS_CONNECTION_STRING is not set.")
            raise ValueError("AZURE_COSMOS_CONNECTION_STRING must be set for MongoDB API.")

    async def connect(self):
        if self.db_type == "cosmosdb_mongo":
            if not MOTOR_AVAILABLE: raise ImportError("motor library not available.")
            if not self.mongodb_connection_string: 
                raise ValueError("MongoDB connection string (AZURE_COSMOS_CONNECTION_STRING) is not set.")
            
            logger.info(f"[DB CONNECT MONGO] Connecting to MongoDB API at {self.mongodb_connection_string.split('@')[-1]}...") # Скрываем часть с кредами
            try:
                self.mongo_client = motor.motor_asyncio.AsyncIOMotorClient(self.mongodb_connection_string)
                # Проверка соединения (опционально, но полезно)
                await self.mongo_client.admin.command('ping') 
                # Имя БД может быть частью строки подключения, но мы также берем его из COSMOS_DATABASE_NAME
                # Если COSMOS_DATABASE_NAME не установлен, motor может использовать 'test' или БД из строки подключения.
                # Явно указываем имя БД из переменной.
                actual_db_name = self.mongodb_database_name
                if not actual_db_name:
                    # Пытаемся извлечь из строки подключения, если она стандартная
                    try:
                        # mongodb://user:pass@host:port/DBNAME?options
                        db_from_uri = self.mongodb_connection_string.split('/')[-1].split('?')[0]
                        if db_from_uri and db_from_uri != self.mongo_client.HOST.split(':')[0]: # Проверка, что это не хост
                            actual_db_name = db_from_uri
                        else:
                            actual_db_name = "scientificdata" # Запасной вариант
                            logger.warning(f"Could not determine DB name from URI, using default: {actual_db_name}")
                    except Exception:
                        actual_db_name = "scientificdata"
                        logger.warning(f"Could not determine DB name from URI, using default: {actual_db_name}")
                
                self.mongodb_database_name = actual_db_name # Обновляем, если извлекли из URI
                self.mongo_db = self.mongo_client[self.mongodb_database_name]
                logger.info(f"[DB CONNECT MONGO] Connected to MongoDB. Database: '{self.mongodb_database_name}'.")
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {e}", exc_info=True)
                self.mongo_client = None # Сбрасываем, если ошибка
                self.mongo_db = None
                raise
        elif self.db_type == "postgresql":
            logger.info(f"[DB CONNECT PGSQL] Connecting to {self.sql_db_url}...")
            try:
                self.sql_connection = await asyncpg.connect(self.sql_db_url)
                logger.info(f"[DB CONNECT PGSQL] Connected.")
            except Exception as e:
                logger.error(f"Failed to connect to PostgreSQL: {e}", exc_info=True)
                raise
        elif self.db_type == "sqlite":
            db_path = self.sql_db_url.replace("sqlite:///", "")
            logger.info(f"[DB CONNECT SQLITE] Connecting to {db_path}...")
            try:
                self.sql_connection = sqlite3.connect(db_path) # sqlite3 не async
                self.sql_connection.row_factory = sqlite3.Row
                logger.info(f"[DB CONNECT SQLITE] Connected.")
            except Exception as e:
                logger.error(f"Failed to connect to SQLite: {e}", exc_info=True)
                raise
        else:
            logger.error(f"Unsupported DB_TYPE: {self.db_type}")
            raise ValueError(f"Unsupported DB_TYPE: {self.db_type}")

    async def disconnect(self):
        if self.mongo_client:
            self.mongo_client.close() # motor client has close()
            self.mongo_client = None
            self.mongo_db = None
            logger.info(f"[DB DISCONNECT MONGO] Connection closed.")
        if self.sql_connection:
            if self.db_type == "postgresql":
                await self.sql_connection.close()
            else: # SQLite
                self.sql_connection.close() # sqlite3 не async
            self.sql_connection = None
            logger.info(f"[DB DISCONNECT {self.db_type.upper()}] Connection closed.")

    async def init_database(self):
        if not ((self.db_type == "cosmosdb_mongo" and self.mongo_db) or 
                  (self.db_type != "cosmosdb_mongo" and self.sql_connection)):
            await self.connect()
        
        logger.info(f"[DB INIT {self.db_type.upper()}] Starting database initialization/verification...")

        if self.db_type == "cosmosdb_mongo":
            if not self.mongo_db: 
                logger.error("[DB INIT MONGO] MongoDB client not initialized.")
                return
            try:
                existing_collections = await self.mongo_db.list_collection_names()
                for collection_name in self.mongo_collections:
                    if collection_name not in existing_collections:
                        logger.info(f"[DB INIT MONGO] Collection '{collection_name}' does not exist. Creating...")
                        logger.info(f"[DB INIT MONGO] Collection '{collection_name}' will be created on first use if not present, or ensure it is pre-created in CosmosDB with appropriate sharding.")
                    else:
                        logger.info(f"[DB INIT MONGO] Collection '{collection_name}' already exists.")
                # TODO: Seed initial data for MongoDB if needed (parsing schema.sql INSERTs)
                # await self._seed_initial_data_mongo()
            except Exception as e:
                logger.error(f"[DB INIT MONGO] Error during MongoDB collection check/creation: {e}", exc_info=True)

        elif self.db_type in ["sqlite", "postgresql"] and os.path.exists("database/schema.sql"):
            logger.info(f"[DB INIT SQL] Found schema file: database/schema.sql for {self.db_type}")
            with open("database/schema.sql", 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            
            if self.db_type == "sqlite":
                try:
                    cursor = self.sql_connection.cursor()
                    cursor.executescript(schema_sql)
                    self.sql_connection.commit()
                    logger.info(f"[DB INIT SQLITE] SQL schema executed successfully.")
                except sqlite3.Error as e_sql:
                    logger.error(f"[DB INIT SQLITE ERROR] executing schema: {e_sql}", exc_info=True)
            elif self.db_type == "postgresql":
                statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
                for i, statement in enumerate(statements):
                    try:
                        await self.sql_connection.execute(statement)
                        logger.debug(f"[DB INIT PGSQL] SQL statement {i+1} executed: {statement[:100]}...")
                    except Exception as e_sql_pg:
                        logger.warning(f"[DB INIT PGSQL WARNING] statement {i+1} ('{statement[:50]}...'): {e_sql_pg}")
                logger.info(f"[DB INIT PGSQL] All SQL statements processed.")
        elif self.db_type in ["sqlite", "postgresql"]:
            logger.error(f"[DB INIT SQL ERROR] Schema file not found at database/schema.sql for {self.db_type}")
        
        logger.info(f"[DB INIT {self.db_type.upper()}] Database initialization/verification finished.")

    # ----- MongoDB Specific Methods (Examples) -----
    async def mongo_insert_one(self, collection_name: str, document: Dict) -> Optional[Any]:
        if self.db_type != "cosmosdb_mongo" or not self.mongo_db: return None
        try:
            collection = self.mongo_db[collection_name]
            result = await collection.insert_one(document)
            logger.debug(f"Inserted document into '{collection_name}' with id {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"Error inserting one into '{collection_name}': {e}", exc_info=True)
            return None

    async def mongo_find_one(self, collection_name: str, query: Dict) -> Optional[Dict]:
        if self.db_type != "cosmosdb_mongo" or not self.mongo_db: return None
        try:
            collection = self.mongo_db[collection_name]
            document = await collection.find_one(query)
            return document
        except Exception as e:
            logger.error(f"Error finding one in '{collection_name}' with query {query}: {e}", exc_info=True)
            return None

    async def mongo_find(self, collection_name: str, query: Dict, limit: int = 0) -> List[Dict]:
        if self.db_type != "cosmosdb_mongo" or not self.mongo_db: return []
        try:
            collection = self.mongo_db[collection_name]
            cursor = collection.find(query)
            if limit > 0:
                cursor = cursor.limit(limit)
            return await cursor.to_list(length=limit if limit > 0 else None) # length=None for all
        except Exception as e:
            logger.error(f"Error finding in '{collection_name}' with query {query}: {e}", exc_info=True)
            return []
    
    async def mongo_update_one(self, collection_name: str, query: Dict, update_doc: Dict, upsert: bool = False) -> int:
        if self.db_type != "cosmosdb_mongo" or not self.mongo_db: return 0
        try:
            collection = self.mongo_db[collection_name]
            result = await collection.update_one(query, {"$set": update_doc}, upsert=upsert)
            logger.debug(f"Updated documents in '{collection_name}'. Matched: {result.matched_count}, Modified: {result.modified_count}, UpsertedId: {result.upserted_id}")
            return result.modified_count or (1 if result.upserted_id else 0)
        except Exception as e:
            logger.error(f"Error updating one in '{collection_name}': {e}", exc_info=True)
            return 0

    async def mongo_delete_one(self, collection_name: str, query: Dict) -> int:
        if self.db_type != "cosmosdb_mongo" or not self.mongo_db: return 0
        try:
            collection = self.mongo_db[collection_name]
            result = await collection.delete_one(query)
            return result.deleted_count
        except Exception as e:
            logger.error(f"Error deleting one in '{collection_name}': {e}", exc_info=True)
            return 0

    # ----- SQL Specific execute_query (remains similar, but uses self.sql_connection) -----
    async def sql_execute_query(self, query: str, params: tuple = None, is_ddl_or_insert_update_delete=False) -> List[Dict]:
        if self.db_type not in ["sqlite", "postgresql"] or not self.sql_connection: 
            logger.error("SQL execute_query called for non-SQL DB or DB not connected.")
            return []
        try:
            if self.db_type == "postgresql":
                if is_ddl_or_insert_update_delete:
                    await self.sql_connection.execute(query, *params if params else [])
                    return [] # No rows to return for DDL/DML usually
                else:
                    rows = await self.sql_connection.fetch(query, *params if params else [])
                    return [dict(row) for row in rows]
            else: # SQLite (synchronous, needs to be run in executor for async context)
                # This part needs careful handling in a fully async app. FastAPI handles it.
                cursor = self.sql_connection.cursor()
                cursor.execute(query, params if params else ())
                if is_ddl_or_insert_update_delete:
                    self.sql_connection.commit()
                    return []
                else:
                    rows = cursor.fetchall()
                    return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"SQL Database error ({self.db_type}) executing query '{query[:100]}...': {e}", exc_info=True)
            return []

    # ----- Adapt existing high-level methods -----
    async def get_astronomical_objects(self, limit: int = 100, object_type: Optional[str] = None, catalog_source: Optional[str] = None) -> List[Dict]:
        collection_name = "astronomical_objects"
        if self.db_type == "cosmosdb_mongo":
            query = {}
            if object_type: query["object_type"] = object_type
            if catalog_source: query["catalog_source"] = catalog_source 
            # Для CosmosDB/MongoDB, shard key (если catalog_source) будет учтен автоматически при запросе к этой коллекции,
            # если коллекция была правильно создана с этим ключом.
            return await self.mongo_find(collection_name, query, limit=limit)
        else: # SQL
            sql = "SELECT * FROM astronomical_objects"
            conditions = []
            sql_params = []
            if object_type:
                conditions.append(f"object_type = { '?' if self.db_type == 'sqlite' else '$%d' % (len(sql_params)+1) }")
                sql_params.append(object_type)
            if catalog_source:
                conditions.append(f"catalog_source = { '?' if self.db_type == 'sqlite' else '$%d' % (len(sql_params)+1) }")
                sql_params.append(catalog_source)
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            sql += f" LIMIT { '?' if self.db_type == 'sqlite' else '$%d' % (len(sql_params)+1) }"
            sql_params.append(limit)
            return await self.sql_execute_query(sql, tuple(sql_params))

    async def get_statistics(self) -> Dict[str, Any]:
        collection_name = "system_statistics"
        if self.db_type == "cosmosdb_mongo":
            stats_list = await self.mongo_find(collection_name, {})
            statistics = {}
            for item in stats_list:
                # Предполагаем, что metric_name это _id или поле 'metric_name'
                metric_name = item.get('metric_name', item.get('_id')) 
                if metric_name:
                    statistics[str(metric_name)] = {
                        'value': item.get('value'),
                        'unit': item.get('unit'),
                        'recorded_at': item.get('recorded_at')
                    }
            return statistics
        else: # SQL
            sql = "SELECT metric_name, metric_value, metric_unit, recorded_at FROM system_statistics"
            rows = await self.sql_execute_query(sql)
            return {row['metric_name']: {'value': row['metric_value'], 'unit': row['metric_unit'], 'recorded_at': row['recorded_at']} for row in rows}
        
    async def cache_api_response(self, cache_key: str, data: Dict, expires_in_hours: int = 24):
        collection_name = "api_cache"
        expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)

        if self.db_type == "cosmosdb_mongo":
            # MongoDB TTL indexes лучше настраивать на уровне коллекции.
            # Здесь мы сохраняем expires_at для ручной проверки или если TTL индекс настроен на это поле.
            document = {
                "_id": cache_key, # Используем cache_key как _id для уникальности и поиска
                "data": data,
                "expires_at": expires_at 
            }
            # upsert=True означает, что если документ с таким _id существует, он будет заменен.
            await self.mongo_update_one(collection_name, {"_id": cache_key}, document, upsert=True)
        else: # SQL
            expires_at_sql = expires_at.isoformat() if self.db_type == "sqlite" else expires_at
            sql = """
            INSERT INTO api_cache (cache_key, cache_value, expires_at)
            VALUES (?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET cache_value = excluded.cache_value, expires_at = excluded.expires_at;
            """ if self.db_type == "sqlite" else """
            INSERT INTO api_cache (cache_key, cache_value, expires_at)
            VALUES ($1, $2, $3)
            ON CONFLICT (cache_key) DO UPDATE SET cache_value = $2, expires_at = $3;
            """
            await self.sql_execute_query(sql, (cache_key, json.dumps(data), expires_at_sql), is_ddl_or_insert_update_delete=True)
        
    async def get_cached_response(self, cache_key: str) -> Optional[Dict]:
        collection_name = "api_cache"
        if self.db_type == "cosmosdb_mongo":
            document = await self.mongo_find_one(collection_name, {"_id": cache_key})
            if document and document.get('expires_at') > datetime.utcnow():
                return document.get('data')
            # Если документ найден, но просрочен, его можно удалить (опционально)
            # if document: await self.mongo_delete_one(collection_name, {"_id": cache_key})
            return None
        else: # SQL
            current_time_sql = datetime.utcnow().isoformat() if self.db_type == "sqlite" else datetime.utcnow()
            sql = "SELECT cache_value FROM api_cache WHERE cache_key = ? AND expires_at > ?"
            params_sql = (cache_key, current_time_sql)
            if self.db_type == "postgresql":
                sql = sql.replace("?", "$1", 1).replace("?", "$2", 1)
            
            result = await self.sql_execute_query(sql, params_sql)
            if result and result[0].get('cache_value'):
                return json.loads(result[0]['cache_value'])
            return None
        
    async def log_search(self, user_id: str, search_type: str, query_text: str, 
                        params: Dict, results_count: int, execution_time_ms: int):
        collection_name = "search_history"
        timestamp = datetime.utcnow()

        if self.db_type == "cosmosdb_mongo":
            document = {
                "user_id": user_id, # Может быть частью shard key, если коллекция так настроена
                "search_type": search_type,
                "search_query": query_text,
                "search_params": params, 
                "results_count": results_count,
                "execution_time_ms": execution_time_ms,
                "created_at": timestamp
            }
            await self.mongo_insert_one(collection_name, document)
        else: # SQL
            created_at_sql = timestamp.isoformat() if self.db_type == "sqlite" else timestamp
            sql = """
            INSERT INTO search_history 
            (user_id, search_type, search_query, search_params, results_count, execution_time_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            if self.db_type == "postgresql":
                 sql = sql.replace("?", "$1", 1).replace("?", "$2", 1).replace("?", "$3", 1).replace("?", "$4", 1).replace("?", "$5", 1).replace("?", "$6", 1).replace("?", "$7", 1)

            await self.sql_execute_query(sql, (
                user_id, search_type, query_text, json.dumps(params), 
                results_count, execution_time_ms, created_at_sql
            ), is_ddl_or_insert_update_delete=True)

db = DatabaseConfig() 