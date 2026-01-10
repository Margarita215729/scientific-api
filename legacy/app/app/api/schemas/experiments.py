"""Pydantic schemas for experiment entity."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ExperimentStatus(str, Enum):
    """Status of experiment execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ExperimentConfig(BaseModel):
    """Configuration for experiment."""
    
    # Cosmology configuration
    cosmology_config_path: str = Field(
        description="Path to cosmology configuration file"
    )
    
    # Quantum configuration
    quantum_config_path: str = Field(
        description="Path to quantum configuration file"
    )
    
    # Graph construction parameters
    n_cosmology_graphs: int = Field(
        default=10,
        ge=1,
        description="Number of cosmology graphs to generate"
    )
    n_quantum_graphs: int = Field(
        default=10,
        ge=1,
        description="Number of quantum graphs to generate"
    )
    
    # Feature extraction parameters
    use_topology_features: bool = Field(
        default=True,
        description="Extract topological features"
    )
    use_spectral_features: bool = Field(
        default=True,
        description="Extract spectral features"
    )
    use_embedding_features: bool = Field(
        default=True,
        description="Extract embedding features"
    )
    
    # ML model parameters
    train_classification: bool = Field(
        default=True,
        description="Train classification models"
    )
    train_similarity: bool = Field(
        default=True,
        description="Train similarity regression models"
    )
    train_clustering: bool = Field(
        default=True,
        description="Perform clustering analysis"
    )
    
    # Distance computation parameters
    compute_gw_distance: bool = Field(
        default=True,
        description="Compute Gromov-Wasserstein distance"
    )
    compute_spectral_distance: bool = Field(
        default=True,
        description="Compute spectral distance"
    )
    compute_distribution_distance: bool = Field(
        default=True,
        description="Compute distribution distance"
    )
    
    # Clustering parameters
    n_clusters: Optional[int] = Field(
        default=None,
        ge=2,
        description="Number of clusters for K-means (if None, uses auto detection)"
    )
    dbscan_eps: float = Field(
        default=0.5,
        gt=0,
        description="DBSCAN epsilon parameter"
    )
    dbscan_min_samples: int = Field(
        default=5,
        ge=1,
        description="DBSCAN min_samples parameter"
    )
    
    # Additional parameters
    test_size: float = Field(
        default=0.2,
        ge=0.1,
        le=0.5,
        description="Test split fraction for classification/regression"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    force_recompute: bool = Field(
        default=False,
        description="Force recomputation of all results (ignore cache)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "cosmology_config_path": "/configs/cosmology_sdss_dr18.yaml",
                "quantum_config_path": "/configs/quantum_heisenberg_2d.yaml",
                "n_cosmology_graphs": 20,
                "n_quantum_graphs": 20,
                "use_topology_features": True,
                "use_spectral_features": True,
                "use_embedding_features": True,
                "train_classification": True,
                "train_similarity": True,
                "train_clustering": True,
                "compute_gw_distance": True,
                "compute_spectral_distance": True,
                "compute_distribution_distance": True,
                "n_clusters": 3,
                "test_size": 0.2,
                "random_state": 42,
                "force_recompute": False
            }
        }


class ExperimentCreate(BaseModel):
    """Request schema for creating a new experiment."""
    
    name: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description="Human-readable experiment name"
    )
    description: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Detailed experiment description"
    )
    config: ExperimentConfig = Field(
        ...,
        description="Experiment configuration parameters"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorizing experiments"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Cosmology vs Quantum Comparison - SDSS DR18",
                "description": "Comparative analysis of cosmological graphs (SDSS DR18 galaxies) vs quantum graphs (2D Heisenberg model)",
                "config": {
                    "cosmology_config_path": "/configs/cosmology_sdss_dr18.yaml",
                    "quantum_config_path": "/configs/quantum_heisenberg_2d.yaml",
                    "n_cosmology_graphs": 20,
                    "n_quantum_graphs": 20
                },
                "tags": ["cosmology", "quantum", "comparison", "sdss_dr18"]
            }
        }


class ExperimentMetadata(BaseModel):
    """Basic experiment metadata (without full results)."""
    
    id: str = Field(
        ...,
        description="Experiment unique identifier (MongoDB ObjectId)"
    )
    name: str = Field(
        ...,
        description="Experiment name"
    )
    description: Optional[str] = Field(
        default=None,
        description="Experiment description"
    )
    status: ExperimentStatus = Field(
        ...,
        description="Current execution status"
    )
    config: ExperimentConfig = Field(
        ...,
        description="Experiment configuration"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Experiment tags"
    )
    created_at: datetime = Field(
        ...,
        description="Creation timestamp"
    )
    updated_at: datetime = Field(
        ...,
        description="Last update timestamp"
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="Execution start timestamp"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Completion timestamp"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if status is FAILED"
    )
    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Execution progress percentage (0-100)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "name": "Cosmology vs Quantum Comparison - SDSS DR18",
                "description": "Comparative analysis of cosmological vs quantum graphs",
                "status": "completed",
                "config": {"cosmology_config_path": "/configs/cosmology_sdss_dr18.yaml"},
                "tags": ["cosmology", "quantum"],
                "created_at": "2025-01-15T10:00:00Z",
                "updated_at": "2025-01-15T10:30:00Z",
                "started_at": "2025-01-15T10:01:00Z",
                "completed_at": "2025-01-15T10:30:00Z",
                "progress": 100.0
            }
        }


class ClassificationMetrics(BaseModel):
    """Classification model evaluation metrics."""
    
    accuracy: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1_score: float = Field(ge=0.0, le=1.0)
    roc_auc: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    cross_val_score_mean: float = Field(ge=0.0, le=1.0)
    cross_val_score_std: float = Field(ge=0.0)
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]


class RegressionMetrics(BaseModel):
    """Regression model evaluation metrics."""
    
    mse: float = Field(ge=0.0)
    rmse: float = Field(ge=0.0)
    mae: float = Field(ge=0.0)
    r2_score: float = Field(le=1.0)
    cross_val_score_mean: float
    cross_val_score_std: float = Field(ge=0.0)


class ClusteringMetrics(BaseModel):
    """Clustering evaluation metrics."""
    
    method: str = Field(description="Clustering method (kmeans, dbscan)")
    n_clusters: int = Field(ge=1, description="Number of clusters found")
    silhouette_score: float = Field(ge=-1.0, le=1.0)
    davies_bouldin_score: float = Field(ge=0.0)
    calinski_harabasz_score: float = Field(ge=0.0)
    cluster_sizes: List[int] = Field(description="Number of graphs per cluster")
    cluster_statistics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Per-cluster feature statistics"
    )


class DistanceMatrixSummary(BaseModel):
    """Summary statistics for distance matrix."""
    
    method: str = Field(description="Distance method (gw, spectral, distribution)")
    matrix_shape: List[int] = Field(description="Shape of distance matrix (n x n)")
    mean_distance: float = Field(ge=0.0)
    median_distance: float = Field(ge=0.0)
    min_distance: float = Field(ge=0.0)
    max_distance: float = Field(ge=0.0)
    std_distance: float = Field(ge=0.0)


class GraphStatistics(BaseModel):
    """Basic statistics for a set of graphs."""
    
    graph_type: str = Field(description="Graph type (cosmology or quantum)")
    n_graphs: int = Field(ge=1)
    avg_nodes: float = Field(ge=0.0)
    avg_edges: float = Field(ge=0.0)
    avg_density: float = Field(ge=0.0, le=1.0)
    avg_clustering_coefficient: float = Field(ge=0.0, le=1.0)


class ExperimentResults(BaseModel):
    """Complete experiment results."""
    
    # Graph statistics
    cosmology_graph_stats: Optional[GraphStatistics] = None
    quantum_graph_stats: Optional[GraphStatistics] = None
    
    # Feature extraction
    n_features_extracted: Optional[int] = Field(
        default=None,
        description="Total number of features extracted per graph"
    )
    feature_names: Optional[List[str]] = Field(
        default=None,
        description="Names of extracted features"
    )
    
    # Classification results
    classification_metrics: Optional[Dict[str, ClassificationMetrics]] = Field(
        default=None,
        description="Metrics per classifier (key: model name)"
    )
    best_classifier: Optional[str] = Field(
        default=None,
        description="Best performing classifier by accuracy"
    )
    
    # Similarity regression results
    similarity_metrics: Optional[Dict[str, RegressionMetrics]] = Field(
        default=None,
        description="Metrics per regressor (key: model name)"
    )
    best_regressor: Optional[str] = Field(
        default=None,
        description="Best performing regressor by R²"
    )
    
    # Clustering results
    clustering_metrics: Optional[Dict[str, ClusteringMetrics]] = Field(
        default=None,
        description="Metrics per clustering method (key: method name)"
    )
    
    # Distance matrices
    distance_matrices: Optional[Dict[str, DistanceMatrixSummary]] = Field(
        default=None,
        description="Summary statistics per distance method"
    )
    
    # Storage paths
    model_paths: Optional[Dict[str, str]] = Field(
        default=None,
        description="Paths to saved model files"
    )
    data_paths: Optional[Dict[str, str]] = Field(
        default=None,
        description="Paths to saved data files (graphs, features, distances)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "cosmology_graph_stats": {
                    "graph_type": "cosmology",
                    "n_graphs": 20,
                    "avg_nodes": 150.5,
                    "avg_edges": 450.2,
                    "avg_density": 0.04,
                    "avg_clustering_coefficient": 0.35
                },
                "n_features_extracted": 42,
                "classification_metrics": {
                    "RandomForest": {
                        "accuracy": 0.95,
                        "precision": 0.94,
                        "recall": 0.96,
                        "f1_score": 0.95,
                        "roc_auc": 0.98
                    }
                },
                "best_classifier": "RandomForest"
            }
        }


class ExperimentResponse(BaseModel):
    """Complete experiment response with metadata and results."""
    
    metadata: ExperimentMetadata = Field(
        ...,
        description="Experiment metadata"
    )
    results: Optional[ExperimentResults] = Field(
        default=None,
        description="Experiment results (only if status is COMPLETED)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "metadata": {
                    "id": "507f1f77bcf86cd799439011",
                    "name": "Experiment 1",
                    "status": "completed",
                    "progress": 100.0
                },
                "results": {
                    "n_features_extracted": 42,
                    "best_classifier": "RandomForest"
                }
            }
        }


class ExperimentListResponse(BaseModel):
    """Response for listing experiments."""
    
    experiments: List[ExperimentMetadata] = Field(
        ...,
        description="List of experiments"
    )
    total: int = Field(
        ...,
        ge=0,
        description="Total number of experiments matching filters"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "experiments": [
                    {
                        "id": "507f1f77bcf86cd799439011",
                        "name": "Experiment 1",
                        "status": "completed"
                    }
                ],
                "total": 1
            }
        }
