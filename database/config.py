import os
import asyncpg
import sqlite3
from typing import Optional, Dict, Any, List
import json
from datetime import datetime, timedelta
import logging

# Azure Cosmos DB SDK
try:
    from azure.cosmos.aio import CosmosClient as CosmosClientAIO
    from azure.cosmos import PartitionKey, exceptions as cosmos_exceptions
    COSMOS_AVAILABLE = True
except ImportError:
    CosmosClientAIO = None
    PartitionKey = None
    cosmos_exceptions = None
    COSMOS_AVAILABLE = False

logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Database configuration and connection manager"""
    
    def __init__(self):
        self.db_type = os.getenv("DB_TYPE", "sqlite")  # sqlite, postgresql, or cosmosdb
        self.db_url = os.getenv("DATABASE_URL", "sqlite:///./scientific_api.db") # For SQL
        
        self.cosmos_endpoint = os.getenv("COSMOS_ENDPOINT")
        self.cosmos_key = os.getenv("COSMOS_KEY")
        self.cosmos_database_name = os.getenv("COSMOS_DATABASE_NAME", "ScientificDB")
        
        self.sql_connection = None # For SQL
        self.cosmos_client = None
        self.cosmos_database_client = None
        
        self.container_clients: Dict[str, Any] = {}

        # Define Cosmos DB containers and their partition keys
        # Adjust partition keys based on your query patterns for better performance/cost
        self.cosmos_containers_config = {
            "users": "/username", # Assuming username is unique and often used in queries
            "api_keys": "/user_id",
            "astronomical_objects": "/catalog_source", # From schema.sql, object_id is unique
            "astronomical_data": "/object_id",   # Assuming data is tied to an astronomical_object
            "search_history": "/user_id",        # If searches are often queried per user
            "ml_analysis_results": "/user_id",   # If results are often queried per user
            "api_cache": "/cache_key",           # cache_key is unique
            "system_statistics": "/metric_name"  # Or a static key like "/singleton" if data is small
        }
        
        # Проверяем доступность Cosmos SDK при инициализации
        if self.db_type == "cosmosdb" and not COSMOS_AVAILABLE:
            logger.error("COSMOS_DB_ENDPOINT and COSMOS_DB_KEY are set, but azure-cosmos library is not available or failed to import.")
            # Можно либо выбросить исключение, либо переключиться на fallback базу данных, если это предусмотрено
            raise ImportError("Azure Cosmos DB SDK is required but not available.")

    async def connect(self):
        """Connect to the configured database type"""
        if self.db_type == "cosmosdb":
            if not self.cosmos_endpoint or not self.cosmos_key:
                logger.error("COSMOS_ENDPOINT and COSMOS_KEY must be set for Cosmos DB")
                raise ValueError("COSMOS_ENDPOINT and COSMOS_KEY must be set for Cosmos DB")
            
            if not COSMOS_AVAILABLE: # Дополнительная проверка на случай, если объект создали до вызова этой ошибки
                 raise ImportError("Azure Cosmos DB SDK is required but not available.")

            logger.info(f"[DB CONNECT {self.db_type.upper()}] Connecting to {self.cosmos_endpoint}...")
            try:
                # Используем асинхронный клиент
                self.cosmos_client = CosmosClientAIO(self.cosmos_endpoint, credential=self.cosmos_key)
                self.cosmos_database_client = self.cosmos_client.get_database_client(self.cosmos_database_name)
                logger.info(f"[DB CONNECT {self.db_type.upper()}] Connected. Database client for '{self.cosmos_database_name}' obtained.")
                # Асинхронно инициализируем клиенты контейнеров
                for container_name in self.cosmos_containers_config.keys():
                    await self._get_container_client(container_name) # await здесь
            except Exception as e:
                logger.error(f"Failed to connect to Cosmos DB or initialize container clients: {e}", exc_info=True)
                raise

        elif self.db_type == "postgresql":
            logger.info(f"[DB CONNECT {self.db_type.upper()}] Connecting to {self.db_url}...")
            try:
                self.sql_connection = await asyncpg.connect(self.db_url)
                logger.info(f"[DB CONNECT {self.db_type.upper()}] Connected.")
            except Exception as e:
                logger.error(f"Failed to connect to PostgreSQL: {e}", exc_info=True)
                raise
        else: # Default to SQLite
            db_path = self.db_url.replace("sqlite:///", "")
            logger.info(f"[DB CONNECT {self.db_type.upper()}] Connecting to {db_path}...")
            try:
                self.sql_connection = sqlite3.connect(db_path)
                self.sql_connection.row_factory = sqlite3.Row
                logger.info(f"[DB CONNECT {self.db_type.upper()}] Connected.")
            except Exception as e:
                logger.error(f"Failed to connect to SQLite: {e}", exc_info=True)
                raise
            
    async def disconnect(self):
        """Disconnect from database"""
        if self.cosmos_client:
            await self.cosmos_client.close() # Асинхронный клиент имеет close()
            self.cosmos_client = None
            self.cosmos_database_client = None
            self.container_clients = {}
            logger.info(f"[DB DISCONNECT {self.db_type.upper()}] Client resources released.")
        if self.sql_connection:
            if self.db_type == "postgresql":
                await self.sql_connection.close()
            else: # SQLite
                self.sql_connection.close()
            self.sql_connection = None
            logger.info(f"[DB DISCONNECT {self.db_type.upper()}] Connection closed.")
                
    async def _get_container_client(self, container_name: str):
        """Асинхронно получает или создает клиент контейнера для Cosmos DB."""
        if self.db_type != "cosmosdb":
            return None

        if not self.cosmos_database_client:
            logger.error(f"[DB COSMOS WARN] database_client not initialized while getting container for {container_name}.")
            # Попытка переподключения или выброс ошибки
            await self.connect() # Попытка переподключиться
            if not self.cosmos_database_client: # Если все еще нет, то ошибка
                 raise RuntimeError("Cosmos DB database client not initialized after attempting to reconnect.")


        if container_name not in self.container_clients:
            logger.info(f"[DB COSMOS] Getting container client for '{container_name}'...")
            try:
                # Получаем клиент существующего контейнера. Создание здесь не происходит.
                # Создание контейнеров должно быть частью init_database.
                self.container_clients[container_name] = self.cosmos_database_client.get_container_client(container_name)
                # Проверим, существует ли контейнер (опционально, но полезно для отладки)
                await self.container_clients[container_name].read() 
                logger.info(f"[DB COSMOS] Container client for '{container_name}' obtained and verified.")
            except cosmos_exceptions.CosmosResourceNotFoundError:
                logger.warning(f"[DB COSMOS WARN] Container '{container_name}' not found. It should be created during init_database.")
                # Не добавляем в кэш, если не найден.
                return None
            except Exception as e:
                logger.error(f"Failed to get/verify container client for '{container_name}': {e}", exc_info=True)
                return None # Или выбросить исключение
        return self.container_clients.get(container_name)

    async def execute_query(self, query: str, params: tuple = None, container_name: Optional[str] = None, is_update_or_delete=False) -> List[Dict]:
        """
        Execute a query. 
        For Cosmos DB: this is a read query. Use `upsert_item` or `delete_item` for writes.
        For SQL: it can be any query.
        `container_name` is required for Cosmos DB queries.
        `is_update_or_delete` is for SQL to know if it needs to commit.
        """
        if not self.sql_connection and not self.cosmos_client:
            await self.connect()
            
        try:
            if self.db_type == "cosmosdb":
                if not container_name:
                    logger.error("container_name is required for Cosmos DB queries")
                    raise ValueError("container_name is required for Cosmos DB queries")
                
                container_client = await self._get_container_client(container_name)
                if not container_client:
                    logger.error(f"[DB COSMOS ERROR] Container '{container_name}' does not exist or client not initialized for query.")
                    return []

                # Для Cosmos DB, `params` должны быть словарем для именованных параметров.
                # Пример: query = "SELECT * FROM c WHERE c.age > @ageParam", params = {"@ageParam": 30}
                # Переделаем SQL-like `params` в формат Cosmos DB.
                cosmos_params = []
                query_for_log = query
                if params:
                    # Это очень грубая адаптация. В идеале, сам запрос должен быть написан под Cosmos DB.
                    # Здесь мы предполагаем, что `?` можно заменить на `@paramN`.
                    # Более надежно - передавать `params` как словарь с именами.
                    for i, value in enumerate(params):
                        param_name = f"@param{i}"
                        query = query.replace("?", param_name, 1) # Заменяем `?` последовательно
                        cosmos_params.append({"name": param_name, "value": value})
                
                logger.info(f"[DB COSMOS QUERY] Executing on '{container_name}': {query_for_log} with mapped params {cosmos_params}")
                
                items = []
                # Используем асинхронный итератор
                async for item in container_client.query_items(
                    query=query,
                    parameters=cosmos_params if cosmos_params else None,
                    enable_cross_partition_query=True # TODO: Оценить необходимость и выключить если PK известен
                ):
                    items.append(item)
                return items

            elif self.db_type == "postgresql":
                if params:
                    result = await self.sql_connection.fetch(query, *params)
                else:
                    result = await self.sql_connection.fetch(query)
                return [dict(row) for row in result]
            else: # SQLite
                # SQLite не поддерживает асинхронность на уровне драйвера sqlite3,
                # поэтому выполняем синхронно в executor'е, если это вызывается из async контекста.
                # Однако, DatabaseConfig сам не управляет event loop, предполагается, что вызывающий код (FastAPI) это делает.
                # Для простоты оставим синхронный вызов, но в реальном async приложении это было бы проблемой.
                # FastAPI обычно запускает синхронный код в отдельном потоке.
                cursor = self.sql_connection.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                if query.strip().upper().startswith("SELECT"):
                    result = cursor.fetchall()
                    return [dict(row) for row in result]
                else: # INSERT, UPDATE, DELETE
                    self.sql_connection.commit()
                    return [{"affected_rows": cursor.rowcount}] # Возвращаем информацию о затронутых строках
        
        except cosmos_exceptions.CosmosHttpResponseError as e:
            logger.error(f"[DB COSMOS ERROR] Cosmos DB HTTP error: {e.status_code}, {e.message}", exc_info=True)
            return [] # Возвращаем пустой список в случае ошибки
        except Exception as e:
            logger.error(f"Database error ({self.db_type}) executing query '{query[:100]}...': {e}", exc_info=True)
            return []

    async def upsert_item(self, container_name: str, item_body: Dict) -> Optional[Dict]:
        """Upserts (creates or replaces) an item in the specified Cosmos DB container."""
        if self.db_type != "cosmosdb":
            logger.error(f"[DB {self.db_type.upper()} ERROR] upsert_item is only for Cosmos DB.")
            # Для SQL баз, нужно использовать execute_query с INSERT OR REPLACE или ON CONFLICT DO UPDATE
            return None
        
        if not self.cosmos_client:
            await self.connect()

        container_client = await self._get_container_client(container_name)
        if not container_client:
            logger.error(f"[DB COSMOS ERROR] Container '{container_name}' not found for upsert.")
            return None
        
        # Убедимся, что у item_body есть 'id', если его нет, но есть ключ, который должен быть id (например, cache_key)
        # Это важно, так как Cosmos DB использует 'id' как уникальный идентификатор.
        if 'id' not in item_body:
            # Пример: если для 'api_cache' ключ это 'cache_key', то он должен стать 'id'
            if container_name == 'api_cache' and 'cache_key' in item_body:
                item_body['id'] = item_body['cache_key']
            elif container_name == 'astronomical_objects' and 'object_id' in item_body:
                 item_body['id'] = item_body['object_id'] # object_id из schema.sql
            # Добавить другие аналогичные правила для других контейнеров, если необходимо
            else:
                logger.warning(f"Attempting to upsert item in '{container_name}' without an 'id' field. This might lead to issues or auto-generated IDs.")
                # Можно сгенерировать id, если это приемлемо: item_body['id'] = str(uuid.uuid4())

        try:
            logger.info(f"[DB COSMOS UPSERT] Upserting item into '{container_name}': id='{item_body.get('id', 'N/A')}' data='{str(item_body)[:100]}...'")
            # Используем асинхронный метод
            response = await container_client.upsert_item(body=item_body)
            logger.info(f"[DB COSMOS UPSERT] Item upserted successfully into '{container_name}'.")
            return response # response это сам документ
        except cosmos_exceptions.CosmosHttpResponseError as e:
            logger.error(f"[DB COSMOS ERROR] Failed to upsert item into '{container_name}': {e.status_code} - {e.message}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"[DB GENERAL ERROR] Unexpected error during upsert into '{container_name}': {e}", exc_info=True)
            return None

    async def delete_item(self, container_name: str, item_id: str, partition_key_value: Any) -> bool:
        """Deletes an item from the specified Cosmos DB container."""
        if self.db_type != "cosmosdb":
            logger.error(f"[DB {self.db_type.upper()} ERROR] delete_item is only for Cosmos DB.")
            return False
        
        if not self.cosmos_client:
            await self.connect()

        container_client = await self._get_container_client(container_name)
        if not container_client:
            logger.error(f"[DB COSMOS ERROR] Container '{container_name}' not found for delete.")
            return False
        
        try:
            logger.info(f"[DB COSMOS DELETE] Deleting item '{item_id}' from '{container_name}' with partition key '{partition_key_value}'.")
            # Используем асинхронный метод
            await container_client.delete_item(item=item_id, partition_key=partition_key_value)
            logger.info(f"[DB COSMOS DELETE] Item '{item_id}' deleted successfully from '{container_name}'.")
            return True
        except cosmos_exceptions.CosmosResourceNotFoundError:
            logger.warning(f"[DB COSMOS DELETE] Item '{item_id}' not found in '{container_name}' for deletion.")
            return False # Не найдено, но не ошибка, если удаление идемпотентно
        except cosmos_exceptions.CosmosHttpResponseError as e:
            logger.error(f"[DB COSMOS ERROR] Failed to delete item '{item_id}' from '{container_name}': {e.status_code} - {e.message}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"[DB GENERAL ERROR] Unexpected error during delete from '{container_name}': {e}", exc_info=True)
            return False

    async def init_database(self):
        """Initialize database: create database and containers for Cosmos DB, or run schema.sql for SQL DBs."""
        if not self.sql_connection and not self.cosmos_client:
             await self.connect()

        logger.info(f"[DB INIT {self.db_type.upper()}] Starting database initialization...")
        
        if self.db_type == "cosmosdb":
            if not self.cosmos_client or not COSMOS_AVAILABLE: # Проверка COSMOS_AVAILABLE
                 logger.error(f"[DB INIT COSMOS ERROR] Cosmos DB client not initialized or SDK not available.")
                 return

            try:
                logger.info(f"[DB INIT COSMOS] Ensuring database '{self.cosmos_database_name}' exists...")
                # create_database_if_not_exists асинхронный у CosmosClientAIO
                self.cosmos_database_client = await self.cosmos_client.create_database_if_not_exists(id=self.cosmos_database_name)
                logger.info(f"[DB INIT COSMOS] Database '{self.cosmos_database_name}' ensured.")
                # self.cosmos_database_client теперь это DatabaseProxy, не нужно переполучать через get_database_client

                for container_name, partition_key_path in self.cosmos_containers_config.items():
                    logger.info(f"[DB INIT COSMOS] Ensuring container '{container_name}' with partition key '{partition_key_path}' exists...")
                    try:
                        # create_container_if_not_exists асинхронный у DatabaseProxy
                        container_proxy = await self.cosmos_database_client.create_container_if_not_exists(
                            id=container_name, 
                            partition_key=PartitionKey(path=partition_key_path)
                        )
                        self.container_clients[container_name] = container_proxy # Сохраняем полученный ContainerProxy
                        logger.info(f"[DB INIT COSMOS] Container '{container_name}' ensured.")
                    except cosmos_exceptions.CosmosHttpResponseError as e:
                        # Если ошибка "уже существует" с другим ключом партиционирования - это проблема.
                        # Код 409 (Conflict) может означать, что контейнер уже существует.
                        # Если он существует с правильным ключом, это нормально.
                        # Если с неправильным - это ошибка конфигурации.
                        if e.status_code == 409:
                             logger.warning(f"[DB INIT COSMOS] Container '{container_name}' already exists. Assuming configuration is correct.")
                             # Попытка получить клиент существующего контейнера
                             self.container_clients[container_name] = self.cosmos_database_client.get_container_client(container_name)
                        else:
                            logger.error(f"[DB INIT COSMOS ERROR] Failed to create/ensure container '{container_name}': {e.status_code} - {e.message}", exc_info=True)
                    except Exception as e_cont:
                        logger.error(f"[DB INIT COSMOS ERROR] Unexpected error for container '{container_name}': {e_cont}", exc_info=True)
                
                # Загрузка начальных данных из schema.sql (только INSERT OR IGNORE)
                await self._seed_initial_data_cosmos()


            except cosmos_exceptions.CosmosHttpResponseError as e:
                logger.error(f"[DB INIT COSMOS ERROR] Cosmos DB HTTP error during init: {e.status_code}, {e.message}", exc_info=True)
            except Exception as e_main:
                logger.error(f"[DB INIT COSMOS ERROR] Unexpected error during Cosmos DB init: {e_main}", exc_info=True)

        elif os.path.exists("database/schema.sql"): # SQL databases
            logger.info(f"[DB INIT {self.db_type.upper()}] Found schema file: database/schema.sql")
            with open("database/schema.sql", 'r', encoding='utf-8') as f: # Указание кодировки
                schema_sql = f.read()
            
            # Для SQLite, выполняем через executescript для обработки нескольких выражений
            if self.db_type == "sqlite" and self.sql_connection:
                try:
                    cursor = self.sql_connection.cursor()
                    cursor.executescript(schema_sql)
                    self.sql_connection.commit()
                    logger.info(f"[DB INIT {self.db_type.upper()}] SQL schema executed successfully using executescript.")
                except sqlite3.Error as e_sql:
                    logger.error(f"[DB INIT {self.db_type.upper()} ERROR] executing schema with executescript: {e_sql}", exc_info=True)
            elif self.db_type == "postgresql" and self.sql_connection: # Для PostgreSQL, выполняем по одному
                statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
                logger.info(f"[DB INIT {self.db_type.upper()}] Found {len(statements)} statements to execute.")
                for i, statement in enumerate(statements):
                    if statement: # Пропускаем пустые стейтменты
                        logger.debug(f"[DB INIT {self.db_type.upper()}] Executing statement {i+1}/{len(statements)}: {statement[:100]}...")
                        try:
                            await self.sql_connection.execute(statement) # asyncpg использует execute
                            logger.debug(f"[DB INIT {self.db_type.upper()}] Statement {i+1} executed successfully.")
                        except Exception as e_sql_pg:
                            # Логируем, но не прерываем, если это ожидаемые ошибки (например, CREATE IF NOT EXISTS)
                            logger.warning(f"[DB INIT {self.db_type.upper()} WARNING] executing statement {i+1}: {statement[:100]}... - Error: {e_sql_pg}")
                logger.info(f"[DB INIT {self.db_type.upper()}] All SQL statements processed for PostgreSQL.")
            else:
                 logger.error(f"[DB INIT {self.db_type.upper()} ERROR] SQL Connection not available or unsupported DB type for schema execution.")
        else:
            logger.error(f"[DB INIT {self.db_type.upper()} ERROR] Schema file not found at database/schema.sql")
        logger.info(f"[DB INIT {self.db_type.upper()}] Database initialization finished.")

    async def _seed_initial_data_cosmos(self):
        """Seeds initial data from schema.sql into Cosmos DB if tables are empty."""
        logger.info("[DB INIT COSMOS] Attempting to seed initial data from schema.sql into Cosmos DB...")
        if not os.path.exists("database/schema.sql"):
            logger.warning("[DB INIT COSMOS] schema.sql not found, skipping initial data seeding for Cosmos DB.")
            return

        with open("database/schema.sql", 'r', encoding='utf-8') as f:
            schema_sql = f.read()

        # Очень упрощенный парсинг INSERT OR IGNORE INTO ... VALUES ...
        # Это не полноценный SQL парсер и может работать не для всех случаев.
        import re
        insert_pattern = re.compile(
            r"INSERT OR IGNORE INTO\s+([a-zA-Z_0-9]+)\s*\(([^)]+)\)\s*VALUES\s*\(([^)]+)\);", 
            re.IGNORECASE | re.MULTILINE | re.DOTALL
        )

        for match in insert_pattern.finditer(schema_sql):
            table_name = match.group(1).strip()
            columns_str = match.group(2).strip()
            values_str = match.group(3).strip()

            if table_name not in self.cosmos_containers_config:
                logger.warning(f"[DB INIT COSMOS SEED] Table '{table_name}' from schema.sql not found in Cosmos DB container config. Skipping.")
                continue

            # Проверка, пуст ли контейнер (очень упрощенная - только один элемент)
            # В реальном приложении лучше проверять более надежно или использовать флаг "seeded"
            try:
                container_client = await self._get_container_client(table_name)
                if not container_client:
                    logger.warning(f"[DB INIT COSMOS SEED] Container client for '{table_name}' not available. Skipping seed.")
                    continue
                
                # Проверяем, есть ли хоть один документ
                query = "SELECT TOP 1 * FROM c"
                items = []
                async for _ in container_client.query_items(query=query, enable_cross_partition_query=True):
                    items.append(_)
                    break 
                if items:
                    logger.info(f"[DB INIT COSMOS SEED] Container '{table_name}' is not empty. Skipping initial data seed for this container.")
                    continue
            except Exception as e_check:
                logger.warning(f"[DB INIT COSMOS SEED] Could not check if container '{table_name}' is empty: {e_check}. Proceeding with caution.")
            

            columns = [col.strip() for col in columns_str.split(',')]
            
            # Обработка значений: это очень хрупко, т.к. SQL значения могут быть строками, числами и т.д.
            # Regex для разбора значений, учитывая строки в одинарных кавычках и числа
            value_parts = []
            current_val = ""
            in_string = False
            for char_idx, char in enumerate(values_str):
                if char == "'" and (char_idx == 0 or values_str[char_idx-1] != '\\'): # Простая проверка на экранирование
                    in_string = not in_string
                    current_val += char
                elif char == "," and not in_string:
                    value_parts.append(current_val.strip())
                    current_val = ""
                else:
                    current_val += char
            value_parts.append(current_val.strip()) # Добавляем последнее значение

            if len(columns) != len(value_parts):
                logger.error(f"[DB INIT COSMOS SEED] Mismatch between columns ({len(columns)}) and values ({len(value_parts)}) for table {table_name}. Statement: {match.group(0)}")
                continue

            item_body = {}
            for i, col_name in enumerate(columns):
                raw_value = value_parts[i]
                # Пытаемся преобразовать тип
                if raw_value.startswith("'") and raw_value.endswith("'"):
                    item_body[col_name] = raw_value[1:-1].replace("''", "'") # Убираем кавычки и двойные кавычки SQL
                elif raw_value.lower() == "null":
                    item_body[col_name] = None
                elif raw_value.lower() == "true":
                    item_body[col_name] = True
                elif raw_value.lower() == "false":
                    item_body[col_name] = False
                else:
                    try:
                        # Пытаемся как число (float или int)
                        if '.' in raw_value:
                            item_body[col_name] = float(raw_value)
                        else:
                            item_body[col_name] = int(raw_value)
                    except ValueError:
                        logger.warning(f"[DB INIT COSMOS SEED] Could not parse value '{raw_value}' for column '{col_name}' in table '{table_name}'. Storing as string.")
                        item_body[col_name] = raw_value # Оставляем как строку, если не удалось распознать

            # Добавляем 'id', если его нет, и если он ожидается из schema.sql
            # Для Cosmos 'id' обязателен. Если в schema.sql есть `id INTEGER PRIMARY KEY AUTOINCREMENT`,
            # то для Cosmos нужно генерировать ID или использовать другой уникальный ключ как 'id'.
            # Для простоты, если 'id' нет, а в columns есть 'object_id' или 'cache_key', используем его.
            if 'id' not in item_body:
                if 'object_id' in item_body:
                    item_body['id'] = str(item_body['object_id'])
                elif 'cache_key' in item_body:
                    item_body['id'] = str(item_body['cache_key'])
                elif 'username' in item_body: # для users
                     item_body['id'] = str(item_body['username'])
                # Добавьте другие правила или генерируйте UUID, если 'id' не предоставлен
                # else:
                #    item_body['id'] = str(uuid.uuid4()) 

            if 'id' not in item_body:
                logger.warning(f"[DB INIT COSMOS SEED] Item for table '{table_name}' does not have an 'id' after parsing. It might be auto-generated by Cosmos DB if not provided. Parsed item: {item_body}")


            logger.info(f"[DB INIT COSMOS SEED] Seeding item into '{table_name}': {item_body}")
            await self.upsert_item(table_name, item_body)
        
        logger.info("[DB INIT COSMOS] Finished attempting to seed initial data.")


    async def get_astronomical_objects(self, limit: int = 100, object_type: Optional[str] = None, catalog_source: Optional[str] = None) -> List[Dict]:
        """Get astronomical objects from database"""
        container_name = "astronomical_objects"
        
        if self.db_type == "cosmosdb":
            query_parts = ["SELECT "]
            if limit > 0:
                 query_parts.append(f"TOP {limit} ")
            query_parts.append("* FROM c")

            conditions = []
            cosmos_params = []
            
            # Ключ партиции для astronomical_objects это catalog_source
            # Если он предоставлен, запрос будет более эффективным
            partition_key_value = None

            if catalog_source:
                conditions.append("c.catalog_source = @catalog_source_param")
                cosmos_params.append({"name": "@catalog_source_param", "value": catalog_source})
                partition_key_value = catalog_source # Это значение ключа партиции

            if object_type:
                conditions.append("c.object_type = @object_type_param")
                cosmos_params.append({"name": "@object_type_param", "value": object_type})
            
            if conditions:
                query_parts.append(" WHERE ")
                query_parts.append(" AND ".join(conditions))
            
            final_query = "".join(query_parts)

            container_client = await self._get_container_client(container_name)
            if not container_client: return []
            
            logger.info(f"[DB COSMOS GET OBJECTS] Querying '{container_name}': {final_query} with params {cosmos_params}")
            try:
                items = []
                # Указываем partition_key если он известен, чтобы избежать cross-partition query или сделать его эффективнее
                query_iterable = container_client.query_items(
                    query=final_query, 
                    parameters=cosmos_params if cosmos_params else None, 
                    partition_key=partition_key_value if partition_key_value else None, # Передаем ключ партиции
                    enable_cross_partition_query=True if not partition_key_value else False # Выключаем, если PK есть
                )
                async for item in query_iterable:
                    items.append(item)
                return items # TOP N уже применен в запросе
            except cosmos_exceptions.CosmosHttpResponseError as e:
                logger.error(f"[DB COSMOS ERROR] get_astronomical_objects: {e.status_code} - {e.message}", exc_info=True)
                return []
        else: # SQL
            query = "SELECT * FROM astronomical_objects"
            sql_params_list = []
            conditions_sql = []
            if object_type:
                conditions_sql.append("object_type = ?")
                sql_params_list.append(object_type)
            if catalog_source:
                conditions_sql.append("catalog_source = ?")
                sql_params_list.append(catalog_source)
            
            if conditions_sql:
                query += " WHERE " + " AND ".join(conditions_sql)

            if limit > 0:
                 query += f" LIMIT ?"
                 sql_params_list.append(limit)
            return await self.execute_query(query, tuple(sql_params_list) if sql_params_list else None)

    async def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        container_name = "system_statistics"
        if self.db_type == "cosmosdb":
            query = "SELECT * FROM c" 
            container_client = await self._get_container_client(container_name)
            if not container_client: return {}
            
            logger.info(f"[DB COSMOS GET STATS] Querying '{container_name}': {query}")
            try:
                statistics = {}
                # Ключ партиции /metric_name. Если мы знаем конкретные метрики, можно передать.
                # Если нет - enable_cross_partition_query.
                async for item in container_client.query_items(query=query, enable_cross_partition_query=True):
                    metric_name = item.get('metric_name', item.get('id')) 
                    if metric_name:
                        statistics[metric_name] = {
                            'value': item.get('metric_value', item.get('value')),
                            'unit': item.get('metric_unit', item.get('unit')),
                            'recorded_at': item.get('recorded_at') # Добавим время записи
                        }
                return statistics
            except cosmos_exceptions.CosmosHttpResponseError as e:
                logger.error(f"[DB COSMOS ERROR] get_statistics: {e.status_code} - {e.message}", exc_info=True)
                return {}
        else: # SQL
            stats_query = "SELECT metric_name, metric_value, metric_unit, recorded_at FROM system_statistics"
            stats_result = await self.execute_query(stats_query)
            statistics = {}
            for row in stats_result:
                statistics[row['metric_name']] = {
                    'value': row['metric_value'],
                    'unit': row['metric_unit'],
                    'recorded_at': row['recorded_at']
                }
            return statistics
        
    async def cache_api_response(self, cache_key: str, data: Dict, expires_in_hours: int = 24):
        """Cache API response"""
        container_name = "api_cache"
        expires_at_dt = datetime.utcnow() + timedelta(hours=expires_in_hours)
        expires_at_iso = expires_at_dt.isoformat() + "Z"


        if self.db_type == "cosmosdb":
            item_body = {
                "id": cache_key, # cache_key это ID и ключ партиции для api_cache
                "cache_key": cache_key, 
                "cache_value": data, 
                "expires_at": expires_at_iso,
                # TTL на уровне контейнера обычно предпочтительнее, но можно и на уровне документа
                # Если TTL контейнера настроен, это поле '_ttl' может быть избыточным или конфликтовать.
                # "_ttl": expires_in_hours * 3600 # TTL в секундах
            }
            await self.upsert_item(container_name, item_body)
        else: # SQL
            expires_at_sql = expires_at_dt if self.db_type == "postgresql" else expires_at_iso
            # Для SQLite, json.dumps обязателен, т.к. нет нативного JSON типа
            query = """
            INSERT INTO api_cache (cache_key, cache_value, expires_at)
            VALUES (?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                cache_value = excluded.cache_value,
                expires_at = excluded.expires_at;
            """ if self.db_type == "sqlite" else """
            INSERT INTO api_cache (cache_key, cache_value, expires_at)
            VALUES ($1, $2, $3)
            ON CONFLICT (cache_key) DO UPDATE SET
                cache_value = $2,
                expires_at = $3;
            """ # PostgreSQL использует $n для плейсхолдеров
            
            params_sql = (cache_key, json.dumps(data), expires_at_sql)
            await self.execute_query(query, params_sql, is_update_or_delete=True)
        
    async def get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Get cached API response, checking for expiration."""
        container_name = "api_cache"
        current_time_iso = datetime.utcnow().isoformat() + "Z"

        if self.db_type == "cosmosdb":
            container_client = await self._get_container_client(container_name)
            if not container_client: return None
            try:
                # Point read по ID и ключу партиции (они совпадают для api_cache)
                logger.debug(f"[DB COSMOS CACHE GET] Reading item '{cache_key}' from '{container_name}' with partition key '{cache_key}'")
                item = await container_client.read_item(item=cache_key, partition_key=cache_key)
                
                # Проверка срока действия TTL уже на стороне Cosmos DB, если _ttl установлен или на контейнере.
                # Но если мы управляем expires_at вручную, то проверяем.
                if item.get('expires_at') and item['expires_at'] > current_time_iso:
                    return item.get('cache_value')
                elif not item.get('expires_at'): # Если нет поля expires_at, считаем валидным (если не используется TTL)
                    logger.warning(f"Cache item '{cache_key}' in '{container_name}' has no 'expires_at' field.")
                    return item.get('cache_value') 
                
                # Если запись просрочена и мы не используем серверный TTL, можно удалить ее здесь.
                # Но это может быть неэффективно при частых чтениях.
                # logger.info(f"Cache item '{cache_key}' expired. Deleting.")
                # await self.delete_item(container_name, cache_key, cache_key)
                return None
            except cosmos_exceptions.CosmosResourceNotFoundError:
                logger.debug(f"[DB COSMOS CACHE GET] Cache miss for key '{cache_key}' (not found).")
                return None
            except Exception as e: # Более общая ошибка, если read_item падает по другой причине
                logger.error(f"[DB COSMOS CACHE GET ERROR] for key '{cache_key}': {e}", exc_info=True)
                return None
        else: # SQL
            current_time_sql = datetime.utcnow() if self.db_type == "postgresql" else current_time_iso
            query = """
            SELECT cache_value FROM api_cache 
            WHERE cache_key = ? AND expires_at > ?
            """
            params_sql = (cache_key, current_time_sql)
            if self.db_type == "postgresql":
                query = query.replace("?", "$1", 1).replace("?", "$2", 1)


            result = await self.execute_query(query, params_sql)
            if result:
                # Для SQLite, json.loads обязателен
                return json.loads(result[0]['cache_value'])
            return None
        
    async def log_search(self, user_id: str, search_type: str, query_text: str, 
                        params: Dict, results_count: int, execution_time_ms: int):
        """Log search history. For Cosmos DB, user_id is the partition key."""
        container_name = "search_history"
        timestamp_iso = datetime.utcnow().isoformat() + "Z"

        if self.db_type == "cosmosdb":
            # Генерируем уникальный ID для каждого лога, например, user_id + timestamp
            # ID документа должен быть уникальным в пределах партиции.
            # Если user_id это ключ партиции, то id может быть просто timestamp или UUID.
            item_id = f"{user_id}_{datetime.utcnow().timestamp()}" # Пример

            item_body = {
                "id": item_id,
                "user_id": user_id, # Это ключ партиции
                "search_type": search_type,
                "search_query": query_text, # Переименовал query в query_text во избежание конфликта с SQL
                "search_params": params, 
                "results_count": results_count,
                "execution_time_ms": execution_time_ms,
                "created_at": timestamp_iso
            }
            await self.upsert_item(container_name, item_body)
        else: # SQL
            insert_query_sql = """
            INSERT INTO search_history 
            (user_id, search_type, search_query, search_params, results_count, execution_time_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            # Для PostgreSQL, плейсхолдеры будут $1, $2, ...
            if self.db_type == "postgresql":
                insert_query_sql = insert_query_sql.replace("?", "$%s") % tuple(range(1, 8))

            await self.execute_query(insert_query_sql, (
                user_id, search_type, query_text, json.dumps(params), 
                results_count, execution_time_ms, timestamp_iso # SQLite сохранит ISO строку как TEXT
            ), is_update_or_delete=True)

# Global database instance
db = DatabaseConfig() 