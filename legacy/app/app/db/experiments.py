"""MongoDB database layer for experiments."""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import DESCENDING

from app.api.schemas.experiments import (
    ExperimentConfig,
    ExperimentCreate,
    ExperimentMetadata,
    ExperimentResults,
    ExperimentResponse,
    ExperimentStatus,
)
from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Collection name
EXPERIMENTS_COLLECTION = "experiments"


def get_mongo_client() -> AsyncIOMotorClient:
    """Get MongoDB async client."""
    settings = get_settings()
    mongo_uri = str(settings.MONGO_URI)
    return AsyncIOMotorClient(mongo_uri)


def get_database() -> AsyncIOMotorDatabase:
    """Get MongoDB database instance."""
    settings = get_settings()
    client = get_mongo_client()
    return client[settings.MONGO_DB_NAME]


async def insert_experiment(experiment_data: ExperimentCreate) -> str:
    """
    Insert a new experiment into MongoDB.
    
    Args:
        experiment_data: Experiment creation data
        
    Returns:
        str: Inserted experiment ID (MongoDB ObjectId as string)
        
    Raises:
        Exception: If database insertion fails
    """
    db = get_database()
    collection = db[EXPERIMENTS_COLLECTION]
    
    now = datetime.utcnow()
    
    document = {
        "name": experiment_data.name,
        "description": experiment_data.description,
        "status": ExperimentStatus.PENDING.value,
        "config": experiment_data.config.model_dump(),
        "tags": experiment_data.tags,
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "completed_at": None,
        "error_message": None,
        "progress": 0.0,
        "results": None,
    }
    
    try:
        result = await collection.insert_one(document)
        experiment_id = str(result.inserted_id)
        logger.info(f"Inserted experiment {experiment_id}: {experiment_data.name}")
        return experiment_id
    except Exception as e:
        logger.error(f"Failed to insert experiment: {e}")
        raise


async def get_experiment(experiment_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve experiment by ID.
    
    Args:
        experiment_id: Experiment ID (MongoDB ObjectId as string)
        
    Returns:
        Optional[Dict]: Experiment document or None if not found
    """
    db = get_database()
    collection = db[EXPERIMENTS_COLLECTION]
    
    try:
        object_id = ObjectId(experiment_id)
    except Exception as e:
        logger.error(f"Invalid experiment_id format: {experiment_id} - {e}")
        return None
    
    try:
        document = await collection.find_one({"_id": object_id})
        if document:
            # Convert ObjectId to string for JSON serialization
            document["id"] = str(document["_id"])
            del document["_id"]
        return document
    except Exception as e:
        logger.error(f"Failed to get experiment {experiment_id}: {e}")
        return None


async def update_experiment_status(
    experiment_id: str,
    status: ExperimentStatus,
    progress: Optional[float] = None,
    error_message: Optional[str] = None,
) -> bool:
    """
    Update experiment status and progress.
    
    Args:
        experiment_id: Experiment ID
        status: New status
        progress: Progress percentage (0-100), optional
        error_message: Error message if status is FAILED, optional
        
    Returns:
        bool: True if update succeeded, False otherwise
    """
    db = get_database()
    collection = db[EXPERIMENTS_COLLECTION]
    
    try:
        object_id = ObjectId(experiment_id)
    except Exception as e:
        logger.error(f"Invalid experiment_id format: {experiment_id} - {e}")
        return False
    
    now = datetime.utcnow()
    update_fields: Dict[str, Any] = {
        "status": status.value,
        "updated_at": now,
    }
    
    if progress is not None:
        update_fields["progress"] = min(max(progress, 0.0), 100.0)
    
    if error_message is not None:
        update_fields["error_message"] = error_message
    
    # Set timestamps based on status
    if status == ExperimentStatus.RUNNING and progress == 0.0:
        update_fields["started_at"] = now
    elif status == ExperimentStatus.COMPLETED:
        update_fields["completed_at"] = now
        update_fields["progress"] = 100.0
    elif status == ExperimentStatus.FAILED:
        update_fields["completed_at"] = now
    
    try:
        result = await collection.update_one(
            {"_id": object_id},
            {"$set": update_fields}
        )
        
        if result.matched_count > 0:
            logger.info(
                f"Updated experiment {experiment_id} status to {status.value}"
                + (f" (progress: {progress}%)" if progress is not None else "")
            )
            return True
        else:
            logger.warning(f"Experiment {experiment_id} not found for status update")
            return False
    except Exception as e:
        logger.error(f"Failed to update experiment {experiment_id} status: {e}")
        return False


async def update_experiment_results(
    experiment_id: str,
    results: ExperimentResults,
) -> bool:
    """
    Update experiment results.
    
    Args:
        experiment_id: Experiment ID
        results: Experiment results
        
    Returns:
        bool: True if update succeeded, False otherwise
    """
    db = get_database()
    collection = db[EXPERIMENTS_COLLECTION]
    
    try:
        object_id = ObjectId(experiment_id)
    except Exception as e:
        logger.error(f"Invalid experiment_id format: {experiment_id} - {e}")
        return False
    
    now = datetime.utcnow()
    
    try:
        result = await collection.update_one(
            {"_id": object_id},
            {
                "$set": {
                    "results": results.model_dump(exclude_none=True),
                    "updated_at": now,
                }
            }
        )
        
        if result.matched_count > 0:
            logger.info(f"Updated experiment {experiment_id} results")
            return True
        else:
            logger.warning(f"Experiment {experiment_id} not found for results update")
            return False
    except Exception as e:
        logger.error(f"Failed to update experiment {experiment_id} results: {e}")
        return False


async def list_experiments(
    status: Optional[ExperimentStatus] = None,
    tags: Optional[List[str]] = None,
    limit: int = 50,
    skip: int = 0,
    sort_by: str = "created_at",
    sort_order: int = -1,  # -1 for descending, 1 for ascending
) -> List[Dict[str, Any]]:
    """
    List experiments with optional filters.
    
    Args:
        status: Filter by status (optional)
        tags: Filter by tags (optional, matches any tag)
        limit: Maximum number of results
        skip: Number of results to skip (for pagination)
        sort_by: Field to sort by
        sort_order: Sort order (-1 descending, 1 ascending)
        
    Returns:
        List[Dict]: List of experiment documents
    """
    db = get_database()
    collection = db[EXPERIMENTS_COLLECTION]
    
    # Build query filter
    query_filter: Dict[str, Any] = {}
    
    if status is not None:
        query_filter["status"] = status.value
    
    if tags:
        query_filter["tags"] = {"$in": tags}
    
    try:
        cursor = collection.find(query_filter)
        
        # Apply sorting
        if sort_order == -1:
            cursor = cursor.sort(sort_by, DESCENDING)
        else:
            cursor = cursor.sort(sort_by, 1)
        
        # Apply pagination
        cursor = cursor.skip(skip).limit(limit)
        
        # Fetch results
        documents = await cursor.to_list(length=limit)
        
        # Convert ObjectId to string
        for doc in documents:
            doc["id"] = str(doc["_id"])
            del doc["_id"]
        
        logger.info(
            f"Listed {len(documents)} experiments "
            f"(status={status}, tags={tags}, skip={skip}, limit={limit})"
        )
        return documents
    except Exception as e:
        logger.error(f"Failed to list experiments: {e}")
        return []


async def count_experiments(
    status: Optional[ExperimentStatus] = None,
    tags: Optional[List[str]] = None,
) -> int:
    """
    Count experiments matching filters.
    
    Args:
        status: Filter by status (optional)
        tags: Filter by tags (optional, matches any tag)
        
    Returns:
        int: Number of experiments matching filters
    """
    db = get_database()
    collection = db[EXPERIMENTS_COLLECTION]
    
    # Build query filter
    query_filter: Dict[str, Any] = {}
    
    if status is not None:
        query_filter["status"] = status.value
    
    if tags:
        query_filter["tags"] = {"$in": tags}
    
    try:
        count = await collection.count_documents(query_filter)
        return count
    except Exception as e:
        logger.error(f"Failed to count experiments: {e}")
        return 0


async def delete_experiment(experiment_id: str) -> bool:
    """
    Delete experiment by ID.
    
    Args:
        experiment_id: Experiment ID
        
    Returns:
        bool: True if deletion succeeded, False otherwise
    """
    db = get_database()
    collection = db[EXPERIMENTS_COLLECTION]
    
    try:
        object_id = ObjectId(experiment_id)
    except Exception as e:
        logger.error(f"Invalid experiment_id format: {experiment_id} - {e}")
        return False
    
    try:
        result = await collection.delete_one({"_id": object_id})
        
        if result.deleted_count > 0:
            logger.info(f"Deleted experiment {experiment_id}")
            return True
        else:
            logger.warning(f"Experiment {experiment_id} not found for deletion")
            return False
    except Exception as e:
        logger.error(f"Failed to delete experiment {experiment_id}: {e}")
        return False


async def create_indexes():
    """Create MongoDB indexes for experiments collection."""
    db = get_database()
    collection = db[EXPERIMENTS_COLLECTION]
    
    try:
        # Index on status for filtering
        await collection.create_index("status")
        
        # Index on tags for filtering
        await collection.create_index("tags")
        
        # Index on created_at for sorting
        await collection.create_index([("created_at", DESCENDING)])
        
        # Compound index for common queries
        await collection.create_index([
            ("status", 1),
            ("created_at", DESCENDING)
        ])
        
        logger.info("Created indexes for experiments collection")
    except Exception as e:
        logger.error(f"Failed to create indexes: {e}")


# Helper function to convert document to ExperimentMetadata
def document_to_metadata(document: Dict[str, Any]) -> ExperimentMetadata:
    """
    Convert MongoDB document to ExperimentMetadata.
    
    Args:
        document: MongoDB document (with 'id' field instead of '_id')
        
    Returns:
        ExperimentMetadata: Pydantic model
    """
    return ExperimentMetadata(
        id=document["id"],
        name=document["name"],
        description=document.get("description"),
        status=ExperimentStatus(document["status"]),
        config=ExperimentConfig(**document["config"]),
        tags=document.get("tags", []),
        created_at=document["created_at"],
        updated_at=document["updated_at"],
        started_at=document.get("started_at"),
        completed_at=document.get("completed_at"),
        error_message=document.get("error_message"),
        progress=document.get("progress", 0.0),
    )


# Helper function to convert document to ExperimentResponse
def document_to_response(document: Dict[str, Any]) -> ExperimentResponse:
    """
    Convert MongoDB document to ExperimentResponse.
    
    Args:
        document: MongoDB document (with 'id' field instead of '_id')
        
    Returns:
        ExperimentResponse: Pydantic model
    """
    metadata = document_to_metadata(document)
    
    results = None
    if document.get("results"):
        results = ExperimentResults(**document["results"])
    
    return ExperimentResponse(
        metadata=metadata,
        results=results,
    )
