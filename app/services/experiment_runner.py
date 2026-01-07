"""Experiment runner service - orchestrates the full ML pipeline."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd

from app.api.schemas.experiments import (
    ClassificationMetrics,
    ClusteringMetrics,
    DistanceMatrixSummary,
    ExperimentConfig,
    ExperimentCreate,
    ExperimentResults,
    ExperimentStatus,
    GraphStatistics,
    RegressionMetrics,
)
from app.core.config import get_settings
from app.db.experiments import (
    get_experiment,
    insert_experiment,
    update_experiment_results,
    update_experiment_status,
)

# ML pipeline imports
from ml.data_cosmology.io import load_cosmology_data
from ml.data_cosmology.preprocessing import preprocess_cosmology_data
from ml.data_quantum.preprocessing import prepare_quantum_system
from ml.features.embeddings import compute_graph_embeddings
from ml.features.feature_table import build_feature_table, save_feature_table
from ml.features.spectral import compute_spectral_features
from ml.features.topology import compute_topological_features
from ml.graphs.cosmology_builder import build_cosmology_graph, save_graph
from ml.graphs.quantum_builder import build_quantum_graph, save_quantum_graph
from ml.metrics.distribution_distance import compute_pairwise_distribution_matrix
from ml.metrics.gw_distance import compute_pairwise_gw_matrix, save_distance_matrix
from ml.metrics.spectral_distance import compute_pairwise_spectral_matrix
from ml.models.classification import (
    evaluate_classifiers,
    prepare_classification_data,
    save_classifier,
    train_classifiers,
)
from ml.models.clustering import (
    cluster_graphs_dbscan,
    cluster_graphs_kmeans,
    evaluate_clustering,
    save_clustering,
)
from ml.models.similarity_regression import (
    build_pairwise_features,
    evaluate_similarity_regressors,
    save_regressor,
    train_similarity_regressors,
)

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Orchestrates the full experiment pipeline:
    1. Data preparation (cosmology + quantum)
    2. Graph construction
    3. Feature extraction
    4. Model training and evaluation
    5. Distance computation
    """

    def __init__(self, experiment_id: str, config: ExperimentConfig):
        """
        Initialize experiment runner.

        Args:
            experiment_id: Experiment ID
            config: Experiment configuration
        """
        self.experiment_id = experiment_id
        self.config = config
        self.settings = get_settings()

        # Data directories
        self.data_root = self.settings.DATA_ROOT
        self.experiment_dir = self.data_root / "experiments" / experiment_id
        self.graphs_dir = self.experiment_dir / "graphs"
        self.cosmology_graphs_dir = self.graphs_dir / "cosmology"
        self.quantum_graphs_dir = self.graphs_dir / "quantum"
        self.features_dir = self.experiment_dir / "features"
        self.models_dir = self.experiment_dir / "models"
        self.distances_dir = self.experiment_dir / "distances"

        # Create directories
        for directory in [
            self.experiment_dir,
            self.graphs_dir,
            self.cosmology_graphs_dir,
            self.quantum_graphs_dir,
            self.features_dir,
            self.models_dir,
            self.distances_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        # Storage for pipeline results
        self.cosmology_graphs: List[nx.Graph] = []
        self.quantum_graphs: List[nx.Graph] = []
        self.feature_table: Optional[pd.DataFrame] = None
        self.results: Dict[str, Any] = {}

    async def run_full_pipeline(self) -> ExperimentResults:
        """
        Run the complete experiment pipeline.

        Returns:
            ExperimentResults: Aggregated results

        Raises:
            Exception: If pipeline execution fails
        """
        logger.info(f"Starting experiment {self.experiment_id}")

        try:
            await update_experiment_status(
                self.experiment_id, ExperimentStatus.RUNNING, progress=0.0
            )

            # Step 1: Data preparation and graph construction (20%)
            await self._step_build_graphs()
            await update_experiment_status(
                self.experiment_id, ExperimentStatus.RUNNING, progress=20.0
            )

            # Step 2: Feature extraction (40%)
            await self._step_extract_features()
            await update_experiment_status(
                self.experiment_id, ExperimentStatus.RUNNING, progress=40.0
            )

            # Step 3: Model training (60%)
            await self._step_train_models()
            await update_experiment_status(
                self.experiment_id, ExperimentStatus.RUNNING, progress=60.0
            )

            # Step 4: Distance computation (80%)
            await self._step_compute_distances()
            await update_experiment_status(
                self.experiment_id, ExperimentStatus.RUNNING, progress=80.0
            )

            # Step 5: Aggregate results (100%)
            results = await self._aggregate_results()
            await update_experiment_results(self.experiment_id, results)
            await update_experiment_status(
                self.experiment_id, ExperimentStatus.COMPLETED, progress=100.0
            )

            logger.info(f"Experiment {self.experiment_id} completed successfully")
            return results

        except Exception as e:
            logger.error(f"Experiment {self.experiment_id} failed: {e}", exc_info=True)
            await update_experiment_status(
                self.experiment_id, ExperimentStatus.FAILED, error_message=str(e)
            )
            raise

    async def _step_build_graphs(self):
        """Step 1: Build cosmology and quantum graphs."""
        logger.info("Building graphs...")

        # Build cosmology graphs
        cosmology_config_path = Path(self.config.cosmology_config_path)
        cosmology_data = load_cosmology_data(cosmology_config_path)
        cosmology_data_clean = preprocess_cosmology_data(cosmology_data)

        for i in range(self.config.n_cosmology_graphs):
            # Sample subset if needed (or use different configs)
            graph = build_cosmology_graph(cosmology_data_clean)
            self.cosmology_graphs.append(graph)
            graph_path = self.cosmology_graphs_dir / f"cosmology_{i}.graphml"
            save_graph(graph, graph_path)
            logger.info(f"Built cosmology graph {i+1}/{self.config.n_cosmology_graphs}")

        # Build quantum graphs
        quantum_config_path = Path(self.config.quantum_config_path)

        for i in range(self.config.n_quantum_graphs):
            quantum_system = prepare_quantum_system(quantum_config_path)
            graph = build_quantum_graph(quantum_system)
            self.quantum_graphs.append(graph)
            graph_path = self.quantum_graphs_dir / f"quantum_{i}.graphml"
            save_quantum_graph(graph, graph_path)
            logger.info(f"Built quantum graph {i+1}/{self.config.n_quantum_graphs}")

        # Compute graph statistics
        self.results["cosmology_stats"] = self._compute_graph_statistics(
            self.cosmology_graphs, "cosmology"
        )
        self.results["quantum_stats"] = self._compute_graph_statistics(
            self.quantum_graphs, "quantum"
        )

        logger.info(
            f"Built {len(self.cosmology_graphs)} cosmology graphs "
            f"and {len(self.quantum_graphs)} quantum graphs"
        )

    async def _step_extract_features(self):
        """Step 2: Extract features from all graphs."""
        logger.info("Extracting features...")

        all_graphs = self.cosmology_graphs + self.quantum_graphs
        graph_ids = [f"cosmo_{i}" for i in range(len(self.cosmology_graphs))] + [
            f"quantum_{i}" for i in range(len(self.quantum_graphs))
        ]
        graph_types = ["cosmology"] * len(self.cosmology_graphs) + ["quantum"] * len(
            self.quantum_graphs
        )

        feature_dicts = []

        for graph, graph_id, graph_type in zip(all_graphs, graph_ids, graph_types):
            features = {"graph_id": graph_id, "graph_type": graph_type}

            # Topology features
            if self.config.use_topology_features:
                topo_features = compute_topological_features(graph)
                features.update({f"topo_{k}": v for k, v in topo_features.items()})

            # Spectral features
            if self.config.use_spectral_features:
                spectral_features = compute_spectral_features(graph)
                features.update(
                    {f"spectral_{k}": v for k, v in spectral_features.items()}
                )

            # Embedding features
            if self.config.use_embedding_features:
                embedding_features = compute_graph_embeddings(graph)
                features.update(
                    {f"embed_{k}": v for k, v in embedding_features.items()}
                )

            feature_dicts.append(features)

        # Build feature table
        self.feature_table = build_feature_table(
            all_graphs,
            graph_ids,
            graph_types,
            include_topology=self.config.use_topology_features,
            include_spectral=self.config.use_spectral_features,
            include_embeddings=self.config.use_embedding_features,
        )

        # Save feature table
        feature_table_path = self.features_dir / "feature_table.csv"
        save_feature_table(self.feature_table, feature_table_path)

        logger.info(
            f"Extracted {len(self.feature_table.columns) - 2} features "
            f"for {len(self.feature_table)} graphs"
        )

    async def _step_train_models(self):
        """Step 3: Train ML models."""
        logger.info("Training ML models...")

        if self.feature_table is None:
            raise ValueError("Feature table not available")

        # Classification
        if self.config.train_classification:
            await self._train_classification()

        # Similarity regression
        if self.config.train_similarity:
            await self._train_similarity()

        # Clustering
        if self.config.train_clustering:
            await self._train_clustering()

    async def _train_classification(self):
        """Train classification models."""
        logger.info("Training classification models...")

        X_train, X_test, y_train, y_test, scaler, feature_cols = (
            prepare_classification_data(
                self.feature_table,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )
        )

        # Train classifiers
        classifiers = train_classifiers(
            X_train, y_train, random_state=self.config.random_state
        )

        # Evaluate classifiers
        metrics_dict = evaluate_classifiers(
            classifiers, X_test, y_test, X_train, y_train
        )

        # Save classifiers
        for name, clf in classifiers.items():
            save_path = self.models_dir / f"classifier_{name}.joblib"
            save_classifier(clf, scaler, feature_cols, save_path)

        # Store results
        self.results["classification_metrics"] = {
            name: ClassificationMetrics(**metrics)
            for name, metrics in metrics_dict.items()
        }

        # Find best classifier
        best_clf = max(metrics_dict.items(), key=lambda x: x[1]["accuracy"])
        self.results["best_classifier"] = best_clf[0]

        logger.info(
            f"Best classifier: {best_clf[0]} (accuracy={best_clf[1]['accuracy']:.4f})"
        )

    async def _train_similarity(self):
        """Train similarity regression models."""
        logger.info("Training similarity regression models...")

        # Build pairwise features (using 'combined' method)
        X_pairs, y_similarity = build_pairwise_features(
            self.feature_table,
            method="combined",
        )

        # Train regressors
        regressors = train_similarity_regressors(
            X_pairs,
            y_similarity,
            random_state=self.config.random_state,
        )

        # Evaluate regressors
        metrics_dict = evaluate_similarity_regressors(regressors, X_pairs, y_similarity)

        # Save regressors
        for name, reg in regressors.items():
            save_path = self.models_dir / f"regressor_{name}.joblib"
            save_regressor(reg, save_path)

        # Store results
        self.results["similarity_metrics"] = {
            name: RegressionMetrics(**metrics) for name, metrics in metrics_dict.items()
        }

        # Find best regressor
        best_reg = max(metrics_dict.items(), key=lambda x: x[1]["r2_score"])
        self.results["best_regressor"] = best_reg[0]

        logger.info(f"Best regressor: {best_reg[0]} (R²={best_reg[1]['r2_score']:.4f})")

    async def _train_clustering(self):
        """Perform clustering analysis."""
        logger.info("Performing clustering analysis...")

        clustering_results = {}

        # K-means clustering
        if self.config.n_clusters:
            X_scaled, feature_cols = cluster_graphs_kmeans(
                self.feature_table,
                n_clusters=self.config.n_clusters,
                random_state=self.config.random_state,
            )

            labels_kmeans = self.feature_table["cluster_id"].values
            metrics_kmeans = evaluate_clustering(X_scaled, labels_kmeans)

            # Save clustering
            save_path = self.models_dir / "clustering_kmeans.joblib"
            save_clustering(
                labels_kmeans,
                {"method": "kmeans", "n_clusters": self.config.n_clusters},
                save_path,
            )

            clustering_results["kmeans"] = ClusteringMetrics(
                method="kmeans",
                n_clusters=self.config.n_clusters,
                **metrics_kmeans,
                cluster_sizes=np.bincount(labels_kmeans).tolist(),
            )

        # DBSCAN clustering
        X_scaled, feature_cols = cluster_graphs_dbscan(
            self.feature_table,
            eps=self.config.dbscan_eps,
            min_samples=self.config.dbscan_min_samples,
        )

        labels_dbscan = self.feature_table["cluster_id"].values

        # Filter out noise points (-1) for metrics
        mask = labels_dbscan != -1
        if mask.sum() > 0:
            metrics_dbscan = evaluate_clustering(X_scaled[mask], labels_dbscan[mask])

            # Save clustering
            save_path = self.models_dir / "clustering_dbscan.joblib"
            save_clustering(
                labels_dbscan,
                {
                    "method": "dbscan",
                    "eps": self.config.dbscan_eps,
                    "min_samples": self.config.dbscan_min_samples,
                },
                save_path,
            )

            n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)

            clustering_results["dbscan"] = ClusteringMetrics(
                method="dbscan",
                n_clusters=n_clusters,
                **metrics_dbscan,
                cluster_sizes=np.bincount(labels_dbscan[labels_dbscan != -1]).tolist(),
            )

        self.results["clustering_metrics"] = clustering_results
        logger.info(
            f"Completed clustering analysis ({len(clustering_results)} methods)"
        )

    async def _step_compute_distances(self):
        """Step 4: Compute distance matrices."""
        logger.info("Computing distance matrices...")

        all_graphs = self.cosmology_graphs + self.quantum_graphs
        graph_ids = [
            *(f"cosmo_{i}" for i in range(len(self.cosmology_graphs))),
            *(f"quantum_{i}" for i in range(len(self.quantum_graphs))),
        ]
        distance_summaries = {}

        # Gromov-Wasserstein distance
        if self.config.compute_gw_distance:
            try:
                gw_matrix = compute_pairwise_gw_matrix(
                    all_graphs,
                    method="degree",
                )
                save_path = self.distances_dir / "gw_distance_matrix.npz"
                save_distance_matrix(gw_matrix, graph_ids, save_path)

                distance_summaries["gromov_wasserstein"] = (
                    self._summarize_distance_matrix(gw_matrix, "gromov_wasserstein")
                )
                logger.info("Computed Gromov-Wasserstein distance matrix")
            except Exception as e:
                logger.warning(f"Failed to compute GW distance: {e}")

        # Spectral distance
        if self.config.compute_spectral_distance:
            spectral_matrix = compute_pairwise_spectral_matrix(
                all_graphs,
                matrix_type="laplacian",
                k=20,
                distance_metric="l2",
            )
            save_path = self.distances_dir / "spectral_distance_matrix.npz"
            np.savez_compressed(
                save_path,
                distance_matrix=spectral_matrix,
                graph_ids=np.array(graph_ids),
            )

            distance_summaries["spectral"] = self._summarize_distance_matrix(
                spectral_matrix, "spectral"
            )
            logger.info("Computed spectral distance matrix")

        # Distribution distance
        if self.config.compute_distribution_distance:
            dist_matrix = compute_pairwise_distribution_matrix(
                all_graphs,
                property_name="degree",
                distance_type="wasserstein",
            )
            save_path = self.distances_dir / "distribution_distance_matrix.npz"
            np.savez_compressed(
                save_path,
                distance_matrix=dist_matrix,
                graph_ids=np.array(graph_ids),
            )

            distance_summaries["distribution"] = self._summarize_distance_matrix(
                dist_matrix, "distribution"
            )
            logger.info("Computed distribution distance matrix")

        self.results["distance_matrices"] = distance_summaries

    async def _aggregate_results(self) -> ExperimentResults:
        """Aggregate all results into ExperimentResults."""
        logger.info("Aggregating experiment results...")

        # Build model paths
        model_paths = {}
        for file_path in self.models_dir.glob("*.joblib"):
            model_paths[file_path.stem] = str(file_path)

        # Build data paths
        data_paths = {
            "feature_table": str(self.features_dir / "feature_table.csv"),
            "cosmology_graphs_dir": str(self.cosmology_graphs_dir),
            "quantum_graphs_dir": str(self.quantum_graphs_dir),
        }
        for file_path in self.distances_dir.glob("*.npz"):
            data_paths[file_path.stem] = str(file_path)

        return ExperimentResults(
            cosmology_graph_stats=self.results.get("cosmology_stats"),
            quantum_graph_stats=self.results.get("quantum_stats"),
            n_features_extracted=(
                len(self.feature_table.columns) - 2
                if self.feature_table is not None
                else None
            ),
            feature_names=(
                [
                    col
                    for col in self.feature_table.columns
                    if col not in ["graph_id", "graph_type"]
                ]
                if self.feature_table is not None
                else None
            ),
            classification_metrics=self.results.get("classification_metrics"),
            best_classifier=self.results.get("best_classifier"),
            similarity_metrics=self.results.get("similarity_metrics"),
            best_regressor=self.results.get("best_regressor"),
            clustering_metrics=self.results.get("clustering_metrics"),
            distance_matrices=self.results.get("distance_matrices"),
            model_paths=model_paths,
            data_paths=data_paths,
        )

    def _compute_graph_statistics(
        self,
        graphs: List[nx.Graph],
        graph_type: str,
    ) -> GraphStatistics:
        """Compute statistics for a list of graphs."""
        n_graphs = len(graphs)

        nodes = [g.number_of_nodes() for g in graphs]
        edges = [g.number_of_edges() for g in graphs]
        densities = [nx.density(g) for g in graphs]

        # Clustering coefficient (average over nodes)
        clusterings = []
        for g in graphs:
            try:
                clustering = nx.average_clustering(g)
                clusterings.append(clustering)
            except:
                clusterings.append(0.0)

        return GraphStatistics(
            graph_type=graph_type,
            n_graphs=n_graphs,
            avg_nodes=float(np.mean(nodes)),
            avg_edges=float(np.mean(edges)),
            avg_density=float(np.mean(densities)),
            avg_clustering_coefficient=float(np.mean(clusterings)),
        )

    def _summarize_distance_matrix(
        self,
        matrix: np.ndarray,
        method: str,
    ) -> DistanceMatrixSummary:
        """Summarize distance matrix statistics."""
        # Extract upper triangle (exclude diagonal)
        n = matrix.shape[0]
        upper_tri_indices = np.triu_indices(n, k=1)
        distances = matrix[upper_tri_indices]

        return DistanceMatrixSummary(
            method=method,
            matrix_shape=list(matrix.shape),
            mean_distance=float(np.mean(distances)),
            median_distance=float(np.median(distances)),
            min_distance=float(np.min(distances)),
            max_distance=float(np.max(distances)),
            std_distance=float(np.std(distances)),
        )


# High-level API functions


async def create_experiment(experiment_data: ExperimentCreate) -> str:
    """
    Create a new experiment.

    Args:
        experiment_data: Experiment creation data

    Returns:
        str: Experiment ID
    """
    experiment_id = await insert_experiment(experiment_data)
    logger.info(f"Created experiment {experiment_id}: {experiment_data.name}")
    return experiment_id


async def run_experiment(experiment_id: str) -> ExperimentResults:
    """
    Run experiment pipeline.

    Args:
        experiment_id: Experiment ID

    Returns:
        ExperimentResults: Experiment results

    Raises:
        ValueError: If experiment not found
        Exception: If pipeline execution fails
    """
    # Get experiment
    document = await get_experiment(experiment_id)
    if not document:
        raise ValueError(f"Experiment {experiment_id} not found")

    # Parse config
    config = ExperimentConfig(**document["config"])

    # Run pipeline
    runner = ExperimentRunner(experiment_id, config)
    results = await runner.run_full_pipeline()

    return results


async def get_experiment_results(experiment_id: str) -> Optional[ExperimentResults]:
    """
    Get experiment results.

    Args:
        experiment_id: Experiment ID

    Returns:
        Optional[ExperimentResults]: Results if experiment is completed, None otherwise
    """
    document = await get_experiment(experiment_id)
    if not document:
        return None

    if document["status"] != ExperimentStatus.COMPLETED.value:
        return None

    if not document.get("results"):
        return None

    return ExperimentResults(**document["results"])
