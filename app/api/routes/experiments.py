"""FastAPI routes for experiment management."""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, status

from app.api.schemas.experiments import (
    ExperimentCreate,
    ExperimentListResponse,
    ExperimentMetadata,
    ExperimentResponse,
    ExperimentResults,
    ExperimentStatus,
)
from app.core.config import get_settings
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
from ml.graphs.base import get_graph_info
from ml.graphs.cosmology_builder import load_graph

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/experiments", tags=["experiments"])


def _serialize_node(node: int, data: dict) -> dict:
    """Convert node attributes into a lightweight JSON-friendly payload."""

    payload = {"id": int(node)}
    for key in ("x", "y", "z", "potential", "system_type", "model_type"):
        if key in data:
            value = data[key]
            if isinstance(value, (int, float, np.integer, np.floating)):
                payload[key] = float(value)
            else:
                payload[key] = str(value)
    return payload


def _graph_to_payload(graph, graph_id: str, max_nodes: int, max_edges: int) -> dict:
    """Serialize graph structure with trimmed nodes/edges for visualization."""

    node_ids = sorted(graph.nodes())
    if max_nodes:
        node_ids = node_ids[:max_nodes]
    node_set = set(node_ids)

    nodes = [_serialize_node(node, graph.nodes[node]) for node in node_ids]

    edges = []
    for u, v, data in graph.edges(data=True):
        if u in node_set and v in node_set:
            edges.append(
                {
                    "source": int(u),
                    "target": int(v),
                    "weight": float(data.get("weight", 1.0)),
                }
            )
            if len(edges) >= max_edges:
                break

    summary = get_graph_info(graph)
    summary["graph_id"] = graph_id

    return {
        "graph_id": graph_id,
        "summary": summary,
        "nodes": nodes,
        "edges": edges,
    }


def _load_distance_matrix(path: Path) -> np.ndarray:
    """Load distance matrix from .npz or .npy file."""

    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        matrix = (
            data["distance_matrix"]
            if "distance_matrix" in data
            else data[data.files[0]]
        )
        data.close()
        return matrix

    return data


def _feature_means_by_type(feature_table_path: Path) -> List[dict]:
    """Compute per-graph-type mean of numeric features."""

    df = pd.read_csv(feature_table_path)
    if "graph_type" not in df.columns:
        raise ValueError("feature_table.csv missing graph_type column")

    numeric_cols = [
        col
        for col in df.columns
        if col not in {"graph_id", "graph_type"}
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    grouped = df.groupby("graph_type")[numeric_cols].mean().reset_index()
    return grouped.to_dict(orient="records")


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

        logger.info(
            f"Queued experiment {experiment_id} for async execution (task_id={task.id})"
        )

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
        default=None, description="Filter by status"
    ),
    tags: Optional[str] = Query(
        default=None, description="Filter by tags (comma-separated)"
    ),
    limit: int = Query(
        default=50, ge=1, le=100, description="Maximum number of results"
    ),
    skip: int = Query(
        default=0, ge=0, description="Number of results to skip (for pagination)"
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
        default=False, description="Also delete experiment data files"
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


@router.get(
    "/{experiment_id}/graphs/{system_type}",
    response_model=dict,
    summary="Get serialized graphs for visualization",
    description="Return lightweight node/edge data for saved graphs of an experiment.",
)
async def get_experiment_graphs_endpoint(
    experiment_id: str,
    system_type: str,
    limit: int = Query(default=3, ge=1, le=20, description="Maximum graphs to return"),
    max_nodes: int = Query(
        default=200, ge=10, le=2000, description="Trim nodes per graph"
    ),
    max_edges: int = Query(
        default=4000, ge=100, le=20000, description="Trim edges per graph"
    ),
) -> dict:
    """Load and serialize saved graphs for client-side plotting."""

    if system_type not in {"cosmology", "quantum"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="system_type must be either 'cosmology' or 'quantum'",
        )

    document = await get_experiment(experiment_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    settings = get_settings()
    graph_dir = (
        Path(settings.DATA_ROOT)
        / "experiments"
        / experiment_id
        / "graphs"
        / system_type
    )

    if not graph_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No saved graphs found for {system_type} in experiment {experiment_id}",
        )

    graph_files = sorted(graph_dir.glob("*.graphml"))
    total_available = len(graph_files)
    if total_available == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No graphml files available for {system_type} in experiment {experiment_id}",
        )

    selected_files = graph_files[:limit]
    graphs_payload = []

    for path in selected_files:
        try:
            graph = load_graph(path)
            payload = _graph_to_payload(
                graph,
                graph_id=path.stem,
                max_nodes=max_nodes,
                max_edges=max_edges,
            )
            payload["source_path"] = str(path)
            graphs_payload.append(payload)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(f"Failed to load graph {path}: {exc}")

    if not graphs_payload:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load graphs for visualization",
        )

    return {
        "experiment_id": experiment_id,
        "system_type": system_type,
        "total_available": total_available,
        "returned": len(graphs_payload),
        "graphs": graphs_payload,
    }


@router.get(
    "/{experiment_id}/plots/{plot_type}",
    response_model=dict,
    summary="Get plot-ready data",
    description="Provide condensed data slices suitable for client-side plotting.",
)
async def get_experiment_plot_data_endpoint(
    experiment_id: str,
    plot_type: str,
    distance_type: Optional[str] = Query(
        default="spectral",
        description="Distance matrix type: spectral, gromov_wasserstein, or distribution",
    ),
    max_size: int = Query(
        default=50,
        ge=5,
        le=200,
        description="Maximum matrix dimension to return for heatmaps",
    ),
) -> dict:
    """Return precomputed data for plots such as distance heatmaps and feature means."""

    if plot_type not in {"distance_heatmap", "feature_means"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="plot_type must be 'distance_heatmap' or 'feature_means'",
        )

    document = await get_experiment(experiment_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    if document.get("status") != ExperimentStatus.COMPLETED.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Experiment {experiment_id} is not completed",
        )

    base_dir = Path(get_settings().DATA_ROOT) / "experiments" / experiment_id

    if plot_type == "distance_heatmap":
        filename_map = {
            "spectral": "spectral_distance_matrix.npz",
            "gromov_wasserstein": "gw_distance_matrix.npz",
            "distribution": "distribution_distance_matrix.npz",
        }
        if distance_type not in filename_map:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid distance_type; use spectral, gromov_wasserstein, or distribution",
            )

        matrix_path = base_dir / "distances" / filename_map[distance_type]
        if not matrix_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Distance matrix not found for type {distance_type}",
            )

        matrix = _load_distance_matrix(matrix_path)
        size = min(max_size, matrix.shape[0])
        matrix_slice = matrix[:size, :size].tolist()

        return {
            "experiment_id": experiment_id,
            "plot_type": plot_type,
            "distance_type": distance_type,
            "matrix_size": [int(v) for v in matrix.shape],
            "returned_size": size,
            "matrix": matrix_slice,
        }

    # feature_means
    feature_table_path = base_dir / "features" / "feature_table.csv"
    if not feature_table_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="feature_table.csv not found for experiment",
        )

    try:
        feature_summary = _feature_means_by_type(feature_table_path)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return {
        "experiment_id": experiment_id,
        "plot_type": plot_type,
        "feature_summary": feature_summary,
        "feature_table_path": str(feature_table_path),
    }
