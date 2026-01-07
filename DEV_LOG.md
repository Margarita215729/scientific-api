# Development Log

## 2026-02-19 | Experiment visualization endpoints

### Tasks Completed
- Saved cosmology and quantum graphs to GraphML during experiment runs and surfaced their directories in results metadata.
- Fixed distance matrix persistence by using the existing GW saver with graph IDs and numpy-based spectral/distribution saves.
- Added FastAPI endpoints to return serialized graphs and plot-ready data (distance heatmaps, feature means) with size safeguards.
- Updated TECHNICAL_PLAN.md to mark the new experiment visualization routes as complete.

### Files Modified
- app/services/experiment_runner.py
- app/api/routes/experiments.py
- TECHNICAL_PLAN.md

### Notes / TODO
- No automated tests were run; run a smoke experiment to ensure graph saving and endpoints work end-to-end once data assets are available.
- Visualization endpoints rely on GraphML outputs under DATA_ROOT/experiments/<id>/graphs.

## 2026-02-18 | Repo hygiene and env prep

### Tasks Completed
- Removed accidentally added CPython 3.11.9 source tree and tarball; updated `.gitignore` to block `Python-*/` and `Python-*.tgz` artifacts.
- Confirmed devcontainer post-create script for git-lfs is present; devcontainer.json change remains staged for commit.
- Updated TECHNICAL_PLAN.md to record the repository hygiene rule.

### Files Modified
- .gitignore
- TECHNICAL_PLAN.md

### Notes / TODO
- Pending: commit `.devcontainer/postCreate.sh` and updated `devcontainer.json` alongside prior packaging changes.
- No tests were run; changes were repository housekeeping only.

## 2025-02-17 | Notebooks: Pipeline & Experiments

### Tasks Completed
- Created and populated notebooks/01_pipeline_implementation.ipynb: config-driven synthetic pipeline, deterministic seeds, graph construction (kNN/grid), spectral/topology features, baseline logistic regression, artifact saves to outputs/pipeline and figures to reports/figures.
- Created notebooks/02_experiments_and_results.ipynb: reproducibility metadata capture, config grid expansion for baseline/ablations/sensitivity, experiment runs with accuracy tracking, sensitivity plot, sanity check, summaries to reports/tables.
- Updated TECHNICAL_PLAN.md to reflect new notebooks (01_pipeline_implementation, 02_experiments_and_results) while keeping remaining planned notebooks open.

### Files Modified
- notebooks/01_pipeline_implementation.ipynb
- notebooks/02_experiments_and_results.ipynb
- TECHNICAL_PLAN.md

### Notes / TODO
- Experiments rely on artifacts from notebook 01; ensure it is run before notebook 02.
- Sensitivity/ablation currently operate on synthetic data; replace with real pipeline outputs once data ingestion is available.
- No tests executed in this iteration due to dependency installation issues noted previously.

## 2026-01-06 | Космологический ingestion и графовый корпус

### Что сделано
- Добавлены пресеты космологии configs/cosmology_presets.yaml (low_z/high_z по инструкции).
- Созданы каталоги data/raw, data/processed, outputs/pipeline, outputs/datasets, reports/figures|tables; .gitignore дополнен под data/outputs/reports и новые форматы.
- Реализованы источники данных:
  - scientific_api/data_sources/cosmology/sdss_dr17_sql.py — тилованные SQL-запросы SkyServer с манифестом.
  - scientific_api/data_sources/cosmology/desi_dr1_lss.py — загрузка DESI DR1 clustering FITS.
- Реализованы маппинг и нормализация:
  - column_map.py, coords.py, schema.py, normalize_points.py с проверками колонок и комовинговыми координатами.
- Добавлен ingestion entrypoint scientific_api/pipeline/cosmology_ingest.py (CLI) с манифестом run_meta.
- Построение корпуса графов и признаков: pipeline/graph_dataset.py (окна 200 Mpc, kNN=12, 500 графов на источник/пресет) и pipeline/features.py (базовые + спектральные признаки, экспорт в reports/tables).
- Devcontainer переключён на python:3.11-bookworm с обязательными pip шагами; requirements дополнен pyarrow,tqdm.

### Файлы
- configs/cosmology_presets.yaml
- scientific_api/data_sources/cosmology/*.py
- scientific_api/data_processing/cosmology/*.py
- scientific_api/pipeline/{cosmology_ingest.py,graph_dataset.py,features.py}
- .devcontainer/devcontainer.json
- requirements.txt
- .gitignore

### Заметки
- Ингест ожидает длительные загрузки SDSS/DESI; сети в контейнере могут быть узким местом.
- Формирование графового корпуса требует достаточно RAM/CPU; параметры k/окно можно уменьшить при отладке.

## 2025-12-19 | Experiment Orchestration & FastAPI Routes - Phase 10 & 11

### Tasks Completed

#### Experiment Entity (Phase 10.1)

1. **Created `/app/api/schemas/experiments.py`** (690 lines):
   - `ExperimentStatus` - Enum for experiment states (pending, running, completed, failed)
   - `ExperimentConfig` - Configuration parameters for experiment
     - Paths to cosmology/quantum configs
     - Graph generation parameters (n_cosmology_graphs, n_quantum_graphs)
     - Feature extraction flags (topology, spectral, embedding)
     - ML model training flags (classification, similarity, clustering)
     - Distance computation flags (GW, spectral, distribution)
     - Clustering parameters (n_clusters, dbscan_eps, dbscan_min_samples)
     - Test/train split, random seed, force_recompute
   - `ExperimentCreate` - Request schema for creating new experiment
     - Name, description, config, tags
   - `ExperimentMetadata` - Experiment metadata without full results
     - ID, status, timestamps, progress, error_message
   - Result schemas:
     - `ClassificationMetrics` - Accuracy, precision, recall, F1, ROC-AUC, confusion matrix
     - `RegressionMetrics` - MSE, RMSE, MAE, R², cross-validation scores
     - `ClusteringMetrics` - Silhouette, Davies-Bouldin, Calinski-Harabasz, cluster sizes
     - `DistanceMatrixSummary` - Mean, median, min, max, std of distance matrix
     - `GraphStatistics` - Avg nodes, edges, density, clustering coefficient
   - `ExperimentResults` - Complete results aggregation
   - `ExperimentResponse` - Combined metadata + results
   - `ExperimentListResponse` - List of experiments with pagination

2. **Created `/app/db/experiments.py`** (408 lines):
   - MongoDB integration using Motor (async driver)
   - `get_mongo_client()` - Get async MongoDB client
   - `get_database()` - Get database instance from settings
   - `insert_experiment()` - Create new experiment document
   - `get_experiment()` - Retrieve experiment by ID
   - `update_experiment_status()` - Update status, progress, timestamps
     - Auto-sets started_at when status → RUNNING
     - Auto-sets completed_at when status → COMPLETED/FAILED
   - `update_experiment_results()` - Store computed results
   - `list_experiments()` - Query with filters (status, tags), pagination, sorting
   - `count_experiments()` - Count matching experiments
   - `delete_experiment()` - Remove experiment by ID
   - `create_indexes()` - Create MongoDB indexes for efficient queries
   - Helper functions:
     - `document_to_metadata()` - Convert MongoDB doc to ExperimentMetadata
     - `document_to_response()` - Convert MongoDB doc to ExperimentResponse

#### Experiment Runner (Phase 10.2)

3. **Created `/app/services/experiment_runner.py`** (672 lines):
   - `ExperimentRunner` class - Full pipeline orchestration
     - Initializes experiment directories (graphs, features, models, distances)
     - Manages experiment state and progress tracking
   
   - **Pipeline Steps**:
     1. `_step_build_graphs()` - Build cosmology and quantum graphs (20% progress)
        - Loads data from config files
        - Preprocesses cosmology data
        - Builds n_cosmology_graphs and n_quantum_graphs
        - Computes graph statistics
     
     2. `_step_extract_features()` - Extract features (40% progress)
        - Topology, spectral, embedding features (configurable)
        - Builds unified feature table
        - Saves feature table to CSV
     
     3. `_step_train_models()` - Train ML models (60% progress)
        - Classification: LogisticRegression, RandomForest, GradientBoosting
        - Similarity regression: Ridge, RandomForest, GradientBoosting
        - Clustering: K-means, DBSCAN
        - Saves trained models to joblib files
        - Records evaluation metrics
     
     4. `_step_compute_distances()` - Compute distance matrices (80% progress)
        - Gromov-Wasserstein distance (if POT available)
        - Spectral distance (Laplacian eigenvalue comparison)
        - Distribution distance (degree distribution)
        - Saves distance matrices to npz files
        - Computes summary statistics
     
     5. `_aggregate_results()` - Build final results (100% progress)
        - Aggregates all metrics, statistics, file paths
        - Returns ExperimentResults
   
   - **High-level API**:
     - `create_experiment()` - Create experiment in DB
     - `run_experiment()` - Execute full pipeline
     - `get_experiment_results()` - Retrieve results if completed
   
   - **Error Handling**:
     - Try-except around each step
     - Updates status to FAILED on error
     - Stores error message in DB
     - Preserves partial results

#### Celery Integration (Phase 10.2 continued)

4. **Created `/app/services/tasks.py`** (254 lines):
   - Celery app configuration with Redis broker/backend
   - Task serialization: JSON
   - Time limits: 2h hard limit, 1h50m soft limit
   - Worker settings: prefetch=1, max_tasks_per_child=10
   
   - **Tasks**:
     - `run_experiment_pipeline_task()` - Async experiment execution
       - Max 2 retries with 5-minute delay
       - Returns serialized ExperimentResults
       - Updates experiment status in DB
     
     - `cleanup_experiment_data_task()` - File cleanup
       - Deletes graphs, features, distances
       - Optional: keep models
       - Returns cleanup statistics
     
     - `batch_run_experiments_task()` - Batch execution
       - Runs multiple experiments sequentially
       - Aggregates success/failure counts
     
     - `cleanup_old_experiments_task()` - Periodic cleanup
       - Deletes experiments older than N days
       - Removes both DB records and files
       - Can be scheduled with Celery Beat

   - OpenAPI models defined for all experiment routes via Pydantic schemas
   - Routes registered under `/api/v1` to appear in generated docs

#### FastAPI Integration (Phase 11)

5. **Created `/app/api/routes/experiments.py`** (423 lines):
   - REST API endpoints under `/experiments`:
     - **POST /** - Create experiment
       - Returns ExperimentMetadata with PENDING status
     
     - **GET /{id}** - Get experiment
       - Returns ExperimentResponse (metadata + results)
     
     - **POST /{id}/run** - Run experiment synchronously
       - Blocks until completion
       - Returns updated ExperimentMetadata
       - Checks for conflicts (already running)
     
     - **POST /{id}/run-async** - Run experiment async via Celery
       - Queues task immediately
       - Returns task_id and status
       - Non-blocking
     
     - **GET /{id}/results** - Get results
       - Only available if status=COMPLETED
       - Returns ExperimentResults
     
     - **GET /** - List experiments
       - Filters: status, tags (comma-separated)
       - Pagination: limit, skip
       - Sorted by created_at descending
       - Returns ExperimentListResponse
     
     - **DELETE /{id}** - Delete experiment
       - Removes from DB
       - Optional: also delete files via Celery task
     
     - **GET /{id}/status** - Get status
       - Returns status, progress, timestamps
   
   - **Error Handling**:
     - 404 for not found
     - 409 for conflicts (already running)
     - 500 for internal errors
     - Detailed error messages
    
6. **Registered experiments router in main app**:
   - Included router under prefix `/api/v1` in `api/index.py`
   - Added startup hook to ensure MongoDB experiment indexes

#### Dependencies Update

6. **Updated `requirements.txt`**:
   - Added `networkx==3.2.1` - Graph algorithms
   - Added `POT==0.9.1` - Python Optimal Transport for GW distance
   - Added `pyyaml==6.0.1` - YAML config parsing
   - Added `redis==5.0.1` - Redis client for Celery
   - Added `celery==5.3.4` - Distributed task queue

### Files Created

- `/app/api/schemas/__init__.py` - Schemas module init
- `/app/api/schemas/experiments.py` - Pydantic schemas (690 lines)
- `/app/db/experiments.py` - MongoDB database layer (408 lines)
- `/app/services/experiment_runner.py` - Pipeline orchestrator (672 lines)
- `/app/services/tasks.py` - Celery tasks (254 lines)
- `/app/api/routes/experiments.py` - FastAPI routes (423 lines)

### Files Modified

- `requirements.txt` - Added networkx, POT, pyyaml, redis, celery
- `TECHNICAL_PLAN.md` - Marked Phase 10.1, 10.2, 11 tasks complete

### Design Decisions

1. **Async-first approach**: All DB operations use Motor async driver for scalability
2. **Progress tracking**: 5-step pipeline with 20% increments for user visibility
3. **Flexible configuration**: ExperimentConfig allows fine-grained control over pipeline
4. **Comprehensive metrics**: Captured metrics from all ML models and distance methods
5. **File persistence**: All intermediate results saved to experiment directory
6. **Error resilience**: Graceful failure handling with error messages in DB
7. **Dual execution modes**: Sync (blocking) and async (Celery) for flexibility
8. **RESTful API**: Standard HTTP methods, status codes, pagination

### Integration Notes

- Uses existing `app/core/config.py` Settings with Pydantic
- Follows MongoDB connection patterns from legacy `database/config.py`
- Imports all ML modules from Phases 1-6 (data, graphs, features, models, metrics)
- Directory structure: `/data/experiments/{experiment_id}/{graphs,features,models,distances}/`
- Celery broker/backend from `settings.get_celery_broker_url()` (defaults to Redis URL)

### API Endpoints Summary

```
POST   /experiments/                    Create experiment
GET    /experiments/                    List experiments (with filters)
GET    /experiments/{id}                Get experiment (metadata + results)
POST   /experiments/{id}/run            Run pipeline (sync)
POST   /experiments/{id}/run-async      Run pipeline (async via Celery)
GET    /experiments/{id}/results        Get results (if completed)
GET    /experiments/{id}/status         Get status and progress
DELETE /experiments/{id}                Delete experiment (optional: files too)
```

### TODO

- [x] Register experiment routes in main FastAPI app
- [x] Create startup script to initialize MongoDB indexes
- [x] Integrate Celery tasks for heavy pipeline steps
- [x] Ensure OpenAPI schema coverage for experiment endpoints
- [ ] Add visualization endpoints (graphs, plots)
- [ ] Implement Celery Beat for periodic tasks
- [ ] Add authentication/authorization
- [ ] Create Prometheus metrics for monitoring
- [ ] Add unit tests for all components
- [ ] Create integration tests for full pipeline
- [ ] Add API documentation (docstrings, examples)
- [ ] Implement rate limiting for API endpoints

### Questions for User

- Should we add authentication/authorization to experiment endpoints?
- Do you want Celery Beat enabled for periodic cleanup tasks?
- Should we expose Celery task status via API (e.g., GET /tasks/{task_id})?
- How to handle concurrent runs of the same experiment (currently blocked with 409)?
- Should we add pagination for experiment listing (currently hardcoded limit=50)?

---

## 2025-12-19 | Geometric Similarity Metrics - Phase 6

### Tasks Completed

#### Experiment Entity (Phase 10.1)

1. **Created `/app/api/schemas/experiments.py`**:
   - `ExperimentStatus` - Enum for experiment states (pending, running, completed, failed)
   - `ExperimentConfig` - Configuration parameters for experiment
     - Paths to cosmology/quantum configs
     - Graph generation parameters (n_cosmology_graphs, n_quantum_graphs)
     - Feature extraction flags (topology, spectral, embedding)
     - ML model training flags (classification, similarity, clustering)
     - Distance computation flags (GW, spectral, distribution)
     - Clustering parameters (n_clusters, dbscan_eps, dbscan_min_samples)
     - Test/train split, random seed, force_recompute
   - `ExperimentCreate` - Request schema for creating new experiment
     - Name, description, config, tags
   - `ExperimentMetadata` - Experiment metadata without full results
     - ID, status, timestamps, progress, error_message
   - Result schemas:
     - `ClassificationMetrics` - Accuracy, precision, recall, F1, ROC-AUC, confusion matrix
     - `RegressionMetrics` - MSE, RMSE, MAE, R², cross-validation scores
     - `ClusteringMetrics` - Silhouette, Davies-Bouldin, Calinski-Harabasz, cluster sizes
     - `DistanceMatrixSummary` - Mean, median, min, max, std of distance matrix
     - `GraphStatistics` - Avg nodes, edges, density, clustering coefficient
   - `ExperimentResults` - Complete results aggregation
   - `ExperimentResponse` - Combined metadata + results
   - `ExperimentListResponse` - List of experiments with pagination

2. **Created `/app/db/experiments.py`**:
   - MongoDB integration using Motor (async driver)
   - `get_mongo_client()` - Get async MongoDB client
   - `get_database()` - Get database instance from settings
   - `insert_experiment()` - Create new experiment document
   - `get_experiment()` - Retrieve experiment by ID
   - `update_experiment_status()` - Update status, progress, timestamps
     - Auto-sets started_at when status → RUNNING
     - Auto-sets completed_at when status → COMPLETED/FAILED
   - `update_experiment_results()` - Store computed results
   - `list_experiments()` - Query with filters (status, tags), pagination, sorting
   - `count_experiments()` - Count matching experiments
   - `delete_experiment()` - Remove experiment by ID
   - `create_indexes()` - Create MongoDB indexes for efficient queries
   - Helper functions:
     - `document_to_metadata()` - Convert MongoDB doc to ExperimentMetadata
     - `document_to_response()` - Convert MongoDB doc to ExperimentResponse

#### Experiment Runner (Phase 10.2)

3. **Created `/app/services/experiment_runner.py`**:
   - `ExperimentRunner` class - Full pipeline orchestration
     - Initializes experiment directories (graphs, features, models, distances)
     - Manages experiment state and progress tracking
   
   - **Pipeline Steps**:
     1. `_step_build_graphs()` - Build cosmology and quantum graphs (20% progress)
        - Loads data from config files
        - Preprocesses cosmology data
        - Builds n_cosmology_graphs and n_quantum_graphs
        - Computes graph statistics
     
     2. `_step_extract_features()` - Extract features (40% progress)
        - Topology, spectral, embedding features (configurable)
        - Builds unified feature table
        - Saves feature table to CSV
     
     3. `_step_train_models()` - Train ML models (60% progress)
        - Classification: LogisticRegression, RandomForest, GradientBoosting
        - Similarity regression: Ridge, RandomForest, GradientBoosting
        - Clustering: K-means, DBSCAN
        - Saves trained models to joblib files
        - Records evaluation metrics
     
     4. `_step_compute_distances()` - Compute distance matrices (80% progress)
        - Gromov-Wasserstein distance (if POT available)
        - Spectral distance (Laplacian eigenvalue comparison)
        - Distribution distance (degree distribution)
        - Saves distance matrices to npz files
        - Computes summary statistics
     
     5. `_aggregate_results()` - Build final results (100% progress)
        - Aggregates all metrics, statistics, file paths
        - Returns ExperimentResults
   
   - **High-level API**:
     - `create_experiment()` - Create experiment in DB
     - `run_experiment()` - Execute full pipeline
     - `get_experiment_results()` - Retrieve results if completed
   
   - **Error Handling**:
     - Try-except around each step
     - Updates status to FAILED on error
     - Stores error message in DB
     - Preserves partial results

### Files Created

- `/app/api/schemas/__init__.py` - Schemas module init
- `/app/api/schemas/experiments.py` - Pydantic schemas (690 lines)
- `/app/db/experiments.py` - MongoDB database layer (408 lines)
- `/app/services/experiment_runner.py` - Pipeline orchestrator (672 lines)

### Design Decisions

1. **Async-first approach**: All DB operations use Motor async driver for scalability
2. **Progress tracking**: 5-step pipeline with 20% increments for user visibility
3. **Flexible configuration**: ExperimentConfig allows fine-grained control over pipeline
4. **Comprehensive metrics**: Captured metrics from all ML models and distance methods
5. **File persistence**: All intermediate results saved to experiment directory
6. **Error resilience**: Graceful failure handling with error messages in DB

### Integration Notes

- Uses existing `app/core/config.py` Settings with Pydantic
- Follows MongoDB connection patterns from legacy `database/config.py`
- Imports all ML modules from Phases 1-6 (data, graphs, features, models, metrics)
- Directory structure: `/data/experiments/{experiment_id}/{graphs,features,models,distances}/`

### TODO

- [ ] Add Celery integration for async pipeline execution
- [ ] Create FastAPI routes (Phase 11)
- [ ] Add POT library to requirements.txt for GW distance
- [ ] Implement experiment visualization endpoints
- [ ] Add unit tests for schemas, DB layer, runner

### Questions

- Should we cache intermediate results between pipeline steps?
- How to handle very large graphs (millions of nodes) - batching strategy?
- Should we support partial pipeline runs (e.g., only feature extraction)?

---

## 2025-12-19 | Geometric Similarity Metrics - Phase 6

### Tasks Completed

#### Gromov-Wasserstein Distance

1. **Created `/ml/metrics/gw_distance.py`**:
   - `graph_to_distance_matrix()` - Convert graph to pairwise distance matrix
     - Supports shortest path distances
     - Handles disconnected graphs (inf → large finite value)
   - `compute_gw_distance()` - Compute GW distance between graphs
     - Wrapper over POT library (Python Optimal Transport)
     - Returns distance and optimal transport plan
     - Supports square_loss and kl_loss
   - `compute_fused_gw_distance()` - Fused GW (structure + node features)
     - Combines structural and feature similarity
     - Alpha parameter controls trade-off
   - `compute_pairwise_gw_matrix()` - Pairwise GW matrix for multiple graphs
   - `save_distance_matrix()` / `load_distance_matrix()` - Persistence

#### Spectral Distance

2. **Created `/ml/metrics/spectral_distance.py`**:
   - `compute_graph_spectrum()` - Extract Laplacian/adjacency eigenvalues
     - Supports normalized and unnormalized matrices
     - Uses sparse eigenvalue solver for efficiency
   - `pad_or_truncate_spectrum()` - Normalize spectrum lengths
   - `compute_laplacian_spectral_distance()` - L2/L1 distance between Laplacian spectra
   - `compute_adjacency_spectral_distance()` - L2/L1 distance between adjacency spectra
   - `compute_spectral_divergence()` - KL/JS divergence between spectral densities
     - Converts spectra to probability distributions via histograms
   - `compute_pairwise_spectral_matrix()` - Pairwise spectral distances

#### Distribution Distance

3. **Created `/ml/metrics/distribution_distance.py`**:
   - `compute_distribution_distance()` - Generic distribution distance
     - Metrics: Wasserstein, KL, JS, KS, Chi-square, L2
     - Handles both sample and histogram inputs
   - `compute_degree_distribution_distance()` - Compare degree distributions
   - `compute_clustering_distribution_distance()` - Compare clustering coefficients
   - `compute_path_length_distribution_distance()` - Compare path length distributions
     - Supports sampling for large graphs
   - `compute_combined_distribution_distance()` - Weighted combination of multiple distributions
   - `compute_pairwise_distribution_matrix()` - Pairwise distribution distances

### Files Created

**Metrics:**
- `/ml/metrics/__init__.py`
- `/ml/metrics/gw_distance.py`
- `/ml/metrics/spectral_distance.py`
- `/ml/metrics/distribution_distance.py`

### Technical Decisions

1. **POT library for GW**: Used Python Optimal Transport for Gromov-Wasserstein computation. Marked as optional dependency (graceful degradation if not installed).

2. **Multiple distance metrics**: Provided diverse metrics for different use cases:
   - GW: Geometric, permutation-invariant
   - Spectral: Global structure via eigenvalues
   - Distribution: Local properties (degree, clustering, paths)

3. **Sparse eigenvalue solvers**: Used scipy.sparse.linalg.eigsh for efficiency on large sparse matrices.

4. **Spectrum normalization**: Pad/truncate spectra to same length for fair comparison across different graph sizes.

5. **Wasserstein distance**: Primary metric for distributions (requires POT or scipy). Earth Mover's Distance is intuitive and robust.

6. **KL vs JS divergence**: Provided both - KL is asymmetric (useful for directed comparison), JS is symmetric and bounded.

7. **Sampling for path lengths**: Large graphs require sampling to avoid O(n²) computation. Sample size configurable.

8. **Combined distribution distance**: Weighted sum allows emphasizing specific graph properties (degree, clustering, paths).

9. **Distance matrix I/O**: Joblib for full metadata, NPZ for compact storage.

10. **Disconnected graphs**: Replace inf distances with 2*max_finite to avoid numerical issues in GW solver.

### Metrics Summary

**Gromov-Wasserstein:**
- Measures: Structural similarity via optimal transport
- Advantages: Permutation-invariant, theoretically grounded
- Complexity: O(n³) - expensive for large graphs
- Use case: Comparing graphs of different sizes with preserved structure

**Spectral Distance:**
- Measures: Eigenvalue distributions (Laplacian/adjacency)
- Advantages: Global structure, fast computation (sparse solvers)
- Complexity: O(n) for k eigenvalues, O(n³) for all
- Use case: Quick structural comparison, spectral graph theory

**Distribution Distance:**
- Measures: Statistical properties (degree, clustering, paths)
- Advantages: Interpretable, captures local structure
- Complexity: O(n) for degree/clustering, O(n²) for paths (with sampling)
- Use case: Domain-specific comparisons, hypothesis testing

### Next Steps (from TECHNICAL_PLAN.md)

**Experiment Orchestration:**
- [ ] Create `/app/api/schemas/experiments.py` - Pydantic models for experiments
- [ ] Create `/app/db/experiments.py` - MongoDB operations
- [ ] Create `/ml/experiments/runner.py` - Experiment orchestration
- [ ] Create `/ml/experiments/tasks.py` - Celery tasks

### TODO

- Add POT to requirements.txt (optional: pip install POT)
- Create example scripts demonstrating distance computation on real graphs
- Add benchmarking utilities for comparing metric computation times
- Consider approximate GW for very large graphs (e.g., sliced Wasserstein)
- Add visualization utilities for transport plans

### Notes

- All metrics handle edge cases (empty graphs, disconnected components)
- GW requires POT library - check POT_AVAILABLE flag before use
- Spectral methods preserve global structure better than local features
- Distribution methods sensitive to graph size - normalize or subsample
- Metrics can be combined via weighted sums for multi-objective comparison

---

## 2025-12-19 | Machine Learning Models - Phase 5

### Tasks Completed

#### Classification Models

1. **Created `/ml/models/classification.py`**:
   - `prepare_classification_data()` - Prepare feature table for classification
     - Automatic label encoding (cosmology=0, quantum=1)
     - NaN/inf handling
     - Train-test split with stratification
     - Feature standardization with StandardScaler
   - `train_classifiers()` - Train multiple classifier models
     - Logistic Regression baseline
     - Random Forest ensemble
     - Gradient Boosting (best performance)
   - `evaluate_classifiers()` - Comprehensive evaluation
     - Accuracy, precision, recall, F1-score, ROC-AUC
     - Cross-validation scores
     - Confusion matrix and classification report
   - `predict_graph_type()` - Predict graph type from features
   - `save_classifier()` / `load_classifier()` - Model persistence with joblib

#### Similarity Regression Models

2. **Created `/ml/models/similarity_regression.py`**:
   - `build_pairwise_features()` - Construct pairwise feature representations
     - Methods: "difference", "concat", "product", "combined"
     - Supports sampling for large datasets (avoid n² explosion)
     - Automatic similarity computation from distance
   - `train_similarity_regressors()` - Train regression models
     - Ridge regression baseline
     - Random Forest regressor
     - Gradient Boosting regressor
   - `evaluate_similarity_regressors()` - Evaluate regression performance
     - MSE, RMSE, MAE, R²
     - Cross-validation R² scores
   - `predict_similarity()` - Predict pairwise graph similarity
   - `save_regressor()` / `load_regressor()` - Model persistence

#### Clustering Models

3. **Created `/ml/models/clustering.py`**:
   - `prepare_clustering_data()` - Prepare features for clustering
   - `cluster_graphs_kmeans()` - K-means clustering
     - Requires specifying number of clusters
     - Works well for spherical clusters
   - `cluster_graphs_dbscan()` - DBSCAN density-based clustering
     - Automatic cluster detection
     - Robust to outliers (noise points labeled -1)
   - `evaluate_clustering()` - Evaluate clustering quality
     - Silhouette score (cohesion and separation)
     - Davies-Bouldin index (cluster similarity)
     - Calinski-Harabasz index (variance ratio)
   - `assign_cluster_labels()` - Add cluster labels to feature table
   - `compute_cluster_statistics()` - Cluster-level summaries
   - `save_clustering()` / `load_clustering()` - Persistence

### Files Created

**ML Models:**
- `/ml/models/__init__.py`
- `/ml/models/classification.py`
- `/ml/models/similarity_regression.py`
- `/ml/models/clustering.py`

### Technical Decisions

1. **Scikit-learn consistency**: All models use scikit-learn API for consistency and interoperability.

2. **Multiple model types**: Provided baseline (Logistic Regression, Ridge) and advanced (Random Forest, Gradient Boosting) models for comparison.

3. **Pairwise features**: Four methods for similarity regression:
   - "difference": |f1 - f2| (most interpretable)
   - "concat": [f1, f2] (preserves full information)
   - "product": f1 * f2 (interaction features)
   - "combined": all above (best performance, highest dimensionality)

4. **Similarity from distance**: Converted distances to similarities using sim = 1/(1+dist) for regression targets.

5. **Cross-validation**: All models support cross-validation for robust performance estimation.

6. **Joblib persistence**: Model saving includes scaler, feature columns, and metadata for reproducibility.

7. **Clustering evaluation**: Multiple metrics (silhouette, Davies-Bouldin, Calinski-Harabasz) since no single metric is universally best.

8. **Noise handling**: DBSCAN labels outliers as -1, evaluation metrics exclude noise points.

### Model Summary

**Classification:**
- Input: Feature table with ~100-150 features per graph
- Output: Binary labels (cosmology vs quantum)
- Best model expected: Gradient Boosting
- Evaluation: F1-score, ROC-AUC, cross-validation accuracy

**Similarity Regression:**
- Input: Pairwise feature differences/combinations
- Output: Similarity scores [0, 1]
- Best model expected: Gradient Boosting with "combined" features
- Evaluation: R², MAE, cross-validation R²

**Clustering:**
- Input: Feature table with ~100-150 features per graph
- Output: Cluster labels
- Methods: K-means (fixed k) and DBSCAN (automatic)
- Evaluation: Silhouette score, Davies-Bouldin index

### Next Steps (from TECHNICAL_PLAN.md)

**Geometric Similarity Metrics:**
- [ ] Create `/ml/metrics/gw_distance.py` - Gromov-Wasserstein distance
- [ ] Create `/ml/metrics/spectral_distance.py` - Spectral distance
- [ ] Create `/ml/metrics/distribution_distance.py` - Distribution comparison

### TODO

- Add hyperparameter tuning utilities (GridSearchCV, RandomizedSearchCV)
- Create example scripts demonstrating full ML pipeline
- Add feature importance analysis for tree-based models
- Consider adding neural network models (MLP) for comparison
- Add utilities for model interpretation (SHAP, permutation importance)

### Notes

- All models include example usage in `if __name__ == "__main__"` blocks
- Logging throughout for transparency
- Standardized error handling for invalid inputs
- Clustering evaluation handles edge cases (single cluster, all noise)
- Pairwise feature construction supports sampling to avoid memory issues

---

## 2025-12-19 | Feature Engineering - Phase 4

### Tasks Completed

#### Topological Features

1. **Created `/ml/features/topology.py`**:
   - `compute_degree_statistics()` - Mean, std, min, max, median of degree distribution
   - `compute_clustering_coefficient()` - Average and std of clustering coefficients
   - `compute_path_statistics()` - Average shortest path and diameter
     - Supports sampling for large graphs
     - Uses largest component for disconnected graphs
   - `compute_centrality_statistics()` - Betweenness and closeness centrality
     - Sampling support for large graphs
   - `compute_connectivity_statistics()` - Node count, edge count, density, components
   - `compute_topology_features()` - Unified function for all topological features

#### Spectral Features

2. **Created `/ml/features/spectral.py`**:
   - `compute_laplacian_spectrum()` - k smallest eigenvalues of Laplacian
   - `compute_adjacency_spectrum()` - k largest eigenvalues of adjacency matrix
   - `compute_spectral_gap()` - Second smallest - smallest eigenvalue
   - `compute_spectral_statistics()` - Mean, std, min, max, median of eigenvalues
   - `compute_spectral_features()` - Complete spectral feature set
     - Laplacian and adjacency spectra
     - Spectral gap and algebraic connectivity
     - Spectral radius
   - `compute_spectral_entropy()` - Entropy of eigenvalue distribution
   - `compute_extended_spectral_features()` - Extended features with entropy, skewness, kurtosis

#### Embeddings

3. **Created `/ml/features/embeddings.py`**:
   - `random_walk()` - Single random walk from a node
   - `generate_random_walks()` - Multiple random walks from all nodes
   - `compute_simple_node_embeddings()` - Simplified DeepWalk-like embeddings
     - Random walk co-occurrence matrix
     - PCA for dimensionality reduction
   - `aggregate_embeddings()` - Graph-level aggregation (mean, max, sum, std, concat)
   - `compute_embedding_features()` - Complete embedding pipeline
   - `compute_positional_encoding()` - Laplacian eigenvector positional encoding
   - `compute_positional_encoding_features()` - Graph-level positional features

#### Feature Table Construction

4. **Created `/ml/features/feature_table.py`**:
   - `compute_all_features()` - Compute all feature types for a single graph
   - `load_graphs_from_directory()` - Load multiple graphs from directory
   - `build_feature_table()` - Build DataFrame from list of graphs
     - Automatically extracts system_type from graph attributes
     - Adds graph_id and metadata columns
   - `save_feature_table()` / `load_feature_table()` - I/O for feature tables
   - `build_feature_table_from_directory()` - Complete pipeline:
     - Load cosmology and quantum graphs
     - Compute all features
     - Save unified table

### Files Created

**Feature Engineering:**
- `/ml/features/topology.py`
- `/ml/features/spectral.py`
- `/ml/features/embeddings.py`
- `/ml/features/feature_table.py`

### Technical Decisions

1. **Sampling for large graphs**: Topological and centrality features support sampling to handle graphs with thousands of nodes efficiently.

2. **Spectral features**: Used scipy.sparse.linalg.eigsh for efficient eigenvalue computation on sparse matrices (Laplacian, adjacency).

3. **Simplified embeddings**: Implemented basic random walk + PCA approach instead of full word2vec. For production, consider using dedicated node2vec library.

4. **Feature aggregation**: Multiple aggregation methods (mean, max, concat) for node-level features to graph-level.

5. **Parquet default**: Default to Parquet format for feature tables (better compression, faster I/O than CSV).

6. **Metadata preservation**: graph_id and system_type automatically included in feature tables.

### Feature Summary

**Topological (15+ features):**
- Degree statistics (mean, std, min, max, median)
- Clustering (avg, std)
- Path statistics (avg shortest path, diameter)
- Centrality (betweenness, closeness - mean, max)
- Connectivity (nodes, edges, density, components)

**Spectral (20+ features):**
- Laplacian eigenvalue statistics
- Adjacency eigenvalue statistics
- Spectral gap, algebraic connectivity
- Spectral radius
- Optional: spectral entropy, skewness, kurtosis
- Optional: raw eigenvalues

**Embeddings (configurable):**
- 64-128 dimensional node embeddings
- Graph-level aggregation
- Optional: Laplacian positional encoding

**Total: ~100-150 features per graph** (depending on configuration)

### Next Steps (from TECHNICAL_PLAN.md)

**Machine Learning Models:**
- [ ] Create `/ml/models/classification.py` - Binary classification (cosmology vs quantum)
- [ ] Create `/ml/models/similarity_regression.py` - Similarity prediction
- [ ] Create `/ml/models/clustering.py` - Graph clustering

### TODO

- Add unit tests for feature computation
- Create example scripts showing full pipeline (data → graphs → features → table)
- Consider adding more graph metrics (modularity, assortativity)
- Optimize embedding computation for very large graphs
- Add feature selection/importance analysis utilities

### Notes

- All feature functions handle edge cases (empty graphs, disconnected graphs)
- Spectral features use sparse matrix operations for efficiency
- Random walk embeddings use fixed seed for reproducibility
- Feature table format compatible with scikit-learn, pandas ML workflows

---

## 2025-12-19 | Graph Construction - Phase 3

### Tasks Completed

#### Base Graph Utilities

1. **Created `/ml/graphs/base.py`**:
   - `normalize_coordinates()` - Normalize node coordinates (unit_cube, unit_sphere, standardize)
   - `normalize_edge_weights()` - Normalize edge weights (minmax, standardize, max)
   - `get_graph_info()` - Extract graph statistics
   - `extract_coordinates_from_graph()` - Get coordinates from node attributes
   - `add_coordinates_to_graph()` - Add coordinate attributes to nodes
   - `create_graph_from_edges()` - Build graph from edge list and weights

#### Cosmology Graph Builder

2. **Created `/ml/graphs/cosmology_builder.py`**:
   - `build_knn_graph()` - Build k-nearest neighbors graph from coordinates
   - `build_cosmology_graph()` - Build graph from galaxy catalog DataFrame
     - Uses sklearn NearestNeighbors for efficient k-NN
     - Supports coordinate normalization
     - Adds system_type='cosmology' attribute
   - `save_graph()` / `load_graph()` - Save/load graphs in multiple formats (GraphML, GEXF, edgelist)
   - `build_and_save_cosmology_graph()` - Complete pipeline function

#### Quantum Graph Builder

3. **Created `/ml/graphs/quantum_builder.py`**:
   - `hamiltonian_to_graph()` - Convert sparse Hamiltonian matrix to graph
     - Edges from non-zero matrix elements
     - Optional threshold for edge inclusion
   - `build_quantum_graph()` - Build graph from quantum system dictionary
     - Flattens 2D grids to node coordinates
     - Adds potential values as node attributes
     - Adds system_type='quantum' attribute
   - `create_reduced_quantum_graph()` - Create downsampled quantum graph
     - Supports uniform and grid sampling methods
     - Useful for large quantum systems
   - `save_quantum_graph()` / `load_quantum_graph()` - I/O functions
   - `build_and_save_quantum_graph()` - Complete pipeline function

#### Graph Consistency Utilities

4. **Created `/ml/graphs/utils.py`**:
   - `downsample_graph()` - Reduce graph to target node count
     - Methods: random, degree-based, betweenness-based
     - Optional connectivity preservation
   - `match_graph_sizes()` - Make multiple graphs same size
   - `scale_edge_weights_to_range()` - Scale weights to specific range
   - `get_graph_statistics()` - Comprehensive graph statistics
   - `compare_graph_properties()` - Compare two graphs
   - `ensure_graph_consistency()` - Full consistency pipeline for graph lists
   - `extract_largest_component()` - Extract largest connected component
   - `relabel_nodes_sequential()` - Relabel nodes to [0, N-1]

### Files Created

**Graph Construction:**
- `/ml/graphs/base.py`
- `/ml/graphs/cosmology_builder.py`
- `/ml/graphs/quantum_builder.py`
- `/ml/graphs/utils.py`

### Technical Decisions

1. **k-NN for cosmology**: Used sklearn's NearestNeighbors with auto algorithm selection for efficient nearest neighbor search in 3D space.

2. **Sparse Hamiltonian to graph**: Only non-zero Hamiltonian elements become edges. Upper triangle only for undirected graphs (avoids duplicates).

3. **Coordinate normalization**: Default to unit_cube [0,1]^D for comparability between cosmology (3D) and quantum (2D) systems.

4. **Edge weight normalization**: Default minmax to [0,1] for consistent scale across different graph types.

5. **GraphML format**: Default save format for full attribute preservation (coordinates, weights, metadata).

6. **Graph consistency**: Separate utilities for matching sizes and normalizing weights allow flexible comparison strategies.

### Implementation Highlights

- **Cosmology graphs**: k-NN with configurable k, distance metric (euclidean, manhattan, etc.)
- **Quantum graphs**: Direct conversion from Hamiltonian sparsity pattern, optional edge threshold
- **Both domains**: System type metadata, normalized coordinates, normalized weights
- **Downsampling**: Multiple methods (random, degree, betweenness) for creating comparable smaller graphs
- **Statistics**: Comprehensive metrics (degree distribution, weight distribution, clustering, components)

### Next Steps (from TECHNICAL_PLAN.md)

**Feature Engineering:**
- [ ] Create `/ml/features/topology.py` - Topological features (degree dist, clustering, diameter)
- [ ] Create `/ml/features/spectral.py` - Spectral features (Laplacian eigenvalues)
- [ ] Create `/ml/features/embeddings.py` - Node embeddings (node2vec or similar)
- [ ] Create `/ml/features/feature_table.py` - Aggregate features into tables

### TODO

- Create example scripts to build graphs from processed data
- Add unit tests for graph construction pipelines
- Verify graph formats work with visualization tools (Gephi, Cytoscape)
- Consider adding more graph metrics (modularity, assortativity, etc.)

### Notes

- All graph builders support coordinate normalization for cross-domain comparison
- Both cosmology and quantum graphs can be downsampled to same size for fair comparison
- GraphML format preserves all attributes but may be slower than binary formats for very large graphs
- Quantum graphs can be very dense (many non-zero Hamiltonian elements) - threshold parameter helps control edge count

---

## 2025-12-19 | Data Layer Implementation - Phase 2

### Tasks Completed

#### Cosmology Data Pipeline

1. **Created `/ml/data_cosmology/io.py`**:
   - `load_sdss_catalog()` - Load SDSS DR17 data from Parquet/CSV
   - `load_desi_catalog()` - Load DESI DR2 data from Parquet/CSV
   - `load_catalog_sample()` - Load and sample from catalogs
   - `save_processed_catalog()` - Save processed data to disk
   - All functions use configurable paths from `DATA_ROOT`

2. **Created `/ml/data_cosmology/preprocessing.py`**:
   - `filter_by_coordinates()` - Filter by RA/Dec ranges
   - `filter_by_redshift()` - Filter by redshift range
   - `compute_comoving_distance()` - Convert redshift to distance (simplified)
   - `ra_dec_z_to_cartesian()` - Convert spherical to 3D Cartesian coordinates
   - `add_cartesian_coordinates()` - Add x, y, z columns to DataFrame
   - `preprocess_catalog()` - Full preprocessing pipeline
   - Uses configurable cosmological parameters (H0, Omega_m, Omega_lambda)

3. **Created configuration files**:
   - `/configs/cosmology_sample.yml` - SDSS DR17 sample configuration
   - `/configs/desi_sample.yml` - DESI DR2 sample configuration
   - Both include filters, sampling parameters, cosmology settings

#### Quantum Data Pipeline

4. **Created `/ml/data_quantum/models.py`**:
   - `harmonic_oscillator_2d_potential()` - 2D harmonic oscillator potential
   - `potential_well_2d()` - Rectangular well with Gaussian perturbations
   - `build_hamiltonian_2d()` - Build sparse Hamiltonian matrix using finite differences
   - `create_grid_2d()` - Generate 2D coordinate grids
   - Sparse matrix representation for efficient storage and computation

5. **Created `/ml/data_quantum/preprocessing.py`**:
   - `generate_harmonic_oscillator()` - Generate complete harmonic oscillator system
   - `generate_potential_well()` - Generate potential well with perturbations
   - `save_quantum_system()` - Save grid, potential, Hamiltonian to disk
   - `load_quantum_system()` - Load quantum system from saved files
   - Uses NPZ format for compressed storage

6. **Created quantum configuration files**:
   - `/configs/quantum/harmonic_oscillator.yml` - Isotropic oscillator
   - `/configs/quantum/harmonic_oscillator_anisotropic.yml` - Different frequencies in x/y
   - `/configs/quantum/potential_well_simple.yml` - Simple rectangular well
   - `/configs/quantum/potential_well_perturbed.yml` - Well with 3 local perturbations

### Files Created

**Cosmology:**
- `/ml/data_cosmology/io.py`
- `/ml/data_cosmology/preprocessing.py`
- `/configs/cosmology_sample.yml`
- `/configs/desi_sample.yml`

**Quantum:**
- `/ml/data_quantum/models.py`
- `/ml/data_quantum/preprocessing.py`
- `/configs/quantum/harmonic_oscillator.yml`
- `/configs/quantum/harmonic_oscillator_anisotropic.yml`
- `/configs/quantum/potential_well_simple.yml`
- `/configs/quantum/potential_well_perturbed.yml`

### Technical Decisions

1. **Simplified cosmology**: Used linear approximation for comoving distance. For production, should integrate proper cosmological distance equation or use astropy.

2. **Sparse Hamiltonian**: Used scipy.sparse for memory-efficient storage of large Hamiltonian matrices (only non-zero elements stored).

3. **Finite difference method**: Standard 5-point stencil for 2D Laplacian in kinetic energy operator.

4. **Configuration-driven**: All parameters externalized to YAML configs for reproducibility.

5. **Modular design**: Separate functions for each step (filtering, coordinate conversion, potential generation, Hamiltonian building) allow flexible composition.

### Next Steps (from TECHNICAL_PLAN.md)

**Graph Construction:**
- [ ] Create `/ml/graphs/base.py` - Base graph utilities and normalization
- [ ] Create `/ml/graphs/cosmology_builder.py` - Build graphs from galaxy catalogs (k-NN)
- [ ] Create `/ml/graphs/quantum_builder.py` - Build graphs from Hamiltonians
- [ ] Create `/ml/graphs/utils.py` - Graph consistency utilities

### TODO

- Add proper cosmological distance calculation (integrate E(z) or use astropy)
- Create example scripts to run data pipelines with configs
- Add unit tests for coordinate transformations
- Consider adding support for FITS format for astronomy data

### Notes

- All quantum models use atomic units (mass=1, hbar=1) by default
- Perturbations in potential wells use Gaussian functions for smooth local variations
- Both pipelines ready for graph construction phase

---

## 2025-12-19 | Infrastructure Setup - Phase 1

### Tasks Completed

1. **Created new project structure** according to TECHNICAL_PLAN.md:
   - `/app` directory with subdirectories: `api`, `core`, `services`, `db`
   - `/ml` directory with subdirectories: `data_cosmology`, `data_quantum`, `graphs`, `features`, `models`, `metrics`
   - `/configs` for experiment configurations
   - `/data` with subdirectories: `raw/`, `processed/`, `models/`
   - `/notebooks` for research notebooks
   - `/scripts` for helper scripts
   - All directories initialized with appropriate `__init__.py` files

2. **Implemented central configuration** (`/app/core/config.py`):
   - Created `Settings` class using `pydantic.Settings`
   - Configured environment variables for:
     - MongoDB: `MONGO_URI`, `MONGO_DB_NAME`
     - Redis: `REDIS_URL`
     - Logging: `LOG_LEVEL`
     - Data paths: `DATA_ROOT`
     - Application settings: `APP_NAME`, `APP_VERSION`, `DEBUG`
     - API configuration: `API_V1_PREFIX`, `CORS_ORIGINS`
     - ML settings: `ML_RANDOM_SEED`, `ML_N_JOBS`
     - Celery settings: `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`
   - Implemented `get_settings()` function with `@lru_cache` for singleton pattern

3. **Setup logging module** (`/app/core/logging.py`):
   - Created `setup_logging()` function for application-wide logging configuration
   - Implemented `get_logger(name)` function for consistent logger creation
   - Configured structured logging format with timestamps and log levels
   - Set third-party library log levels to reduce noise

4. **Created `.env.example` template**:
   - Comprehensive template with all required environment variables
   - Organized into sections: MongoDB, Redis, Logging, Data Storage, Application, ML, Celery
   - Added comments and examples for each variable
   - Included placeholders for future external data sources (SDSS, DESI)

5. **Updated Docker configuration**:
   - **Dockerfile**:
     - Confirmed Python 3.11-slim base image
     - Removed unnecessary dependencies (sqlite3, postgresql-client)
     - Updated directories: removed `/app/galaxy_data`, `/app/models`; added `/app/data`
     - Added `DATA_ROOT` environment variable
   - **docker-compose.yml**:
     - Added MongoDB 7.0 service with health checks
     - Added Redis 7-alpine service with health checks
     - Reconfigured main API service to use local build instead of remote image
     - Removed old Azure/Cosmos DB environment variables
     - Added all new environment variables from `.env.example`
     - Configured service dependencies with health check conditions
     - Added Celery worker service (optional, with profile)
     - Created persistent volumes for MongoDB and Redis data

### Files Modified

- Created: `/app/core/config.py`
- Created: `/app/core/logging.py`
- Created: `.env.example`
- Modified: `Dockerfile`
- Modified: `docker-compose.yml`
- Created: Multiple `__init__.py` files in new directory structure

### Technical Decisions

1. **MongoDB instead of Cosmos DB**: Switched to standard MongoDB for local development and consistency
2. **Redis for caching and Celery**: Single Redis instance for both application caching and Celery message broker
3. **Multi-stage Docker build**: Kept existing pattern for smaller production images
4. **Celery as optional service**: Added Celery worker with profile flag to enable only when needed
5. **Data directory structure**: Organized `/data` with clear separation: `raw`, `processed`, `models`

### Next Steps (from TECHNICAL_PLAN.md)

**Data Layer - Cosmology:**
- [ ] Create `/ml/data_cosmology/io.py` for loading SDSS DR17 and DESI DR2 data
- [ ] Create `/ml/data_cosmology/preprocessing.py` for filtering and coordinate conversion
- [ ] Add configuration file `/configs/cosmology_sample.yml`

**Data Layer - Quantum:**
- [ ] Create `/ml/data_quantum/models.py` with quantum system implementations
- [ ] Create `/ml/data_quantum/preprocessing.py` for Hamiltonian construction
- [ ] Add configuration files in `/configs/quantum/`

### TODO

- Update `requirements.txt` or create `pyproject.toml` with all necessary dependencies (pydantic-settings, motor, redis, celery, etc.)
- Create MongoDB connection module in `/app/db/`
- Verify that existing `/api` endpoints remain compatible with new structure
- Plan migration strategy for existing utility modules in `/utils`

### Notes

- Old `/api`, `/utils`, `/database` directories remain intact for backward compatibility
- Will gradually migrate functionality to new structure as we implement features from TECHNICAL_PLAN.md
- External integrations (ADS, arXiv, NASA, CERN) preserved in `/utils` for future use
