"""FastAPI routes for experiment management."""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse

from app.api.schemas.experiments import (
    ExperimentCreate,
    ExperimentListResponse,
    ExperimentMetadata,
    ExperimentResponse,
    ExperimentResults,
    ExperimentStatus,
)
from app.db.experiments import (
    count_experiments,
    delete_experiment,
    document_to_metadata,
    document_to_response,
    get_experiment,
    list_experiments,
)
from app.services.experiment_runner import (
    create_experiment,
    get_experiment_results,
    run_experiment,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post(
    "/",
    response_model=ExperimentMetadata,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new experiment",
    description="Create a new experiment with specified configuration. The experiment will be in PENDING status.",
)
async def create_experiment_endpoint(
    experiment_data: ExperimentCreate,
) -> ExperimentMetadata:
    """
    Create a new experiment.
    
    Args:
        experiment_data: Experiment creation payload
        
    Returns:
        ExperimentMetadata: Created experiment metadata
        
    Raises:
        HTTPException: If creation fails
    """
    try:
        experiment_id = await create_experiment(experiment_data)
        
        # Retrieve created experiment
        document = await get_experiment(experiment_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve created experiment {experiment_id}",
            )
        
        metadata = document_to_metadata(document)
        logger.info(f"Created experiment {experiment_id}: {experiment_data.name}")
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to create experiment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create experiment: {str(e)}",
        )


@router.get(
    "/{experiment_id}",
    response_model=ExperimentResponse,
    summary="Get experiment by ID",
    description="Retrieve experiment metadata and results (if completed).",
)
async def get_experiment_endpoint(
    experiment_id: str,
) -> ExperimentResponse:
    """
    Get experiment by ID.
    
    Args:
        experiment_id: Experiment ID (MongoDB ObjectId)
        
    Returns:
        ExperimentResponse: Experiment metadata and results
        
    Raises:
        HTTPException: If experiment not found
    """
    document = await get_experiment(experiment_id)
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )
    
    response = document_to_response(document)
    return response


@router.post(
    "/{experiment_id}/run",
    response_model=ExperimentMetadata,
    summary="Run experiment pipeline",
    description=(
        "Trigger the full experiment pipeline execution. "
        "This is a synchronous operation that may take a long time. "
        "For async execution, use the Celery task endpoint."
    ),
)
async def run_experiment_endpoint(
    experiment_id: str,
) -> ExperimentMetadata:
    """
    Run experiment pipeline synchronously.
    
    Args:
        experiment_id: Experiment ID
        
    Returns:
        ExperimentMetadata: Updated experiment metadata
        
    Raises:
        HTTPException: If experiment not found or pipeline fails
    """
    # Check if experiment exists
    document = await get_experiment(experiment_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )
    
    # Check if already running
    if document["status"] == ExperimentStatus.RUNNING.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Experiment {experiment_id} is already running",
        )
    
    try:
        # Run pipeline
        await run_experiment(experiment_id)
        
        # Retrieve updated experiment
        document = await get_experiment(experiment_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve experiment {experiment_id} after run",
            )
        
        metadata = document_to_metadata(document)
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to run experiment {experiment_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run experiment: {str(e)}",
        )


@router.post(
    "/{experiment_id}/run-async",
    response_model=dict,
    summary="Run experiment pipeline asynchronously (Celery)",
    description="Queue the experiment pipeline for async execution via Celery.",
)
async def run_experiment_async_endpoint(
    experiment_id: str,
) -> dict:
    """
    Run experiment pipeline asynchronously via Celery.
    
    Args:
        experiment_id: Experiment ID
        
    Returns:
        dict: Celery task info
        
    Raises:
        HTTPException: If experiment not found or queueing fails
    """
    # Check if experiment exists
    document = await get_experiment(experiment_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )
    
    # Check if already running
    if document["status"] == ExperimentStatus.RUNNING.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Experiment {experiment_id} is already running",
        )
    
    try:
        # Queue Celery task
        from app.services.tasks import run_experiment_pipeline_task
        
        task = run_experiment_pipeline_task.delay(experiment_id)
        
        logger.info(f"Queued experiment {experiment_id} for async execution (task_id={task.id})")
        
        return {
            "experiment_id": experiment_id,
            "task_id": task.id,
            "status": "queued",
            "message": "Experiment queued for execution",
        }
        
    except Exception as e:
        logger.error(f"Failed to queue experiment {experiment_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue experiment: {str(e)}",
        )


@router.get(
    "/{experiment_id}/results",
    response_model=ExperimentResults,
    summary="Get experiment results",
    description="Retrieve aggregated experiment results. Only available if experiment is completed.",
)
async def get_experiment_results_endpoint(
    experiment_id: str,
) -> ExperimentResults:
    """
    Get experiment results.
    
    Args:
        experiment_id: Experiment ID
        
    Returns:
        ExperimentResults: Aggregated results
        
    Raises:
        HTTPException: If experiment not found or not completed
    """
    results = await get_experiment_results(experiment_id)
    
    if results is None:
        document = await get_experiment(experiment_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found",
            )
        
        status_val = document["status"]
        if status_val != ExperimentStatus.COMPLETED.value:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Experiment {experiment_id} is not completed (status: {status_val})",
            )
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Results not available for experiment {experiment_id}",
        )
    
    return results


@router.get(
    "/",
    response_model=ExperimentListResponse,
    summary="List experiments",
    description="List experiments with optional filters and pagination.",
)
async def list_experiments_endpoint(
    status: Optional[ExperimentStatus] = Query(
        default=None,
        description="Filter by status"
    ),
    tags: Optional[str] = Query(
        default=None,
        description="Filter by tags (comma-separated)"
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=100,
        description="Maximum number of results"
    ),
    skip: int = Query(
        default=0,
        ge=0,
        description="Number of results to skip (for pagination)"
    ),
) -> ExperimentListResponse:
    """
    List experiments with filters.
    
    Args:
        status: Filter by status (optional)
        tags: Filter by tags (optional, comma-separated)
        limit: Maximum number of results
        skip: Number of results to skip
        
    Returns:
        ExperimentListResponse: List of experiments
    """
    # Parse tags
    tag_list = None
    if tags:
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
    
    # Query experiments
    documents = await list_experiments(
        status=status,
        tags=tag_list,
        limit=limit,
        skip=skip,
        sort_by="created_at",
        sort_order=-1,  # Descending (newest first)
    )
    
    # Count total
    total = await count_experiments(status=status, tags=tag_list)
    
    # Convert to metadata
    experiments = [document_to_metadata(doc) for doc in documents]
    
    return ExperimentListResponse(
        experiments=experiments,
        total=total,
    )


@router.delete(
    "/{experiment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete experiment",
    description="Delete experiment from database (does not delete data files).",
)
async def delete_experiment_endpoint(
    experiment_id: str,
    delete_files: bool = Query(
        default=False,
        description="Also delete experiment data files"
    ),
):
    """
    Delete experiment.
    
    Args:
        experiment_id: Experiment ID
        delete_files: Whether to also delete data files
        
    Raises:
        HTTPException: If experiment not found or deletion fails
    """
    success = await delete_experiment(experiment_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )
    
    # Delete files if requested
    if delete_files:
        try:
            from app.services.tasks import cleanup_experiment_data_task
            cleanup_experiment_data_task.delay(experiment_id, keep_models=False)
            logger.info(f"Queued file cleanup for experiment {experiment_id}")
        except Exception as e:
            logger.warning(f"Failed to queue file cleanup for {experiment_id}: {e}")
    
    logger.info(f"Deleted experiment {experiment_id}")
    return None


@router.get(
    "/{experiment_id}/status",
    response_model=dict,
    summary="Get experiment status",
    description="Get current execution status and progress.",
)
async def get_experiment_status_endpoint(
    experiment_id: str,
) -> dict:
    """
    Get experiment status and progress.
    
    Args:
        experiment_id: Experiment ID
        
    Returns:
        dict: Status information
        
    Raises:
        HTTPException: If experiment not found
    """
    document = await get_experiment(experiment_id)
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )
    
    return {
        "experiment_id": experiment_id,
        "status": document["status"],
        "progress": document.get("progress", 0.0),
        "started_at": document.get("started_at"),
        "completed_at": document.get("completed_at"),
        "error_message": document.get("error_message"),
    }
