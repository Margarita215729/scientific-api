# TECHNICAL_PLAN.md — Extended Technical Plan and Checklist for AI Assistant

This document defines how you, an AI coding assistant running inside VS Code (Copilot Chat or similar), must work on the `scientific-api` project.

Your primary goal:  
**Transform the repository into a reproducible backend service and ML pipeline that supports the Bachelor thesis on graph-based comparison of cosmological and quantum systems.**

---

## 0. Execution Rules for the AI Assistant

When you work on this project, always follow these rules:

1. **Single source of truth**
   - Treat this `TECHNICAL_PLAN.md` as the main specification.
   - Before any change, reread the relevant section to ensure alignment.
   - Do not invent new major features that are not derived from this plan.

2. **Atomic, incremental changes**
   - Work in small, coherent steps.
   - For each step:
     - Identify the subsection in this plan.
     - Implement only the tasks from that subsection.
     - Update or create code and documentation that directly relate to that subsection.

3. **Clean code and repo hygiene**
   - Remove dead code, unused modules and obsolete files when you replace functionality.
   - Keep imports minimal and relevant.
   - Respect the chosen stack and avoid unnecessary extra dependencies.

4. **Formatting and style**
   - Format Python code with **Black** (line length 88).
   - Fix style issues reported by **Ruff**.
   - Keep type hints consistent where they already exist.
   - Follow PEP 8 semantics.

5. **Runtime and environment**
   - Assume **Python 3.11**.
   - Assume dependencies are managed through `pyproject.toml` and a virtual environment.
   - Use environment variables for secrets and external services, never hard-code them.

6. **No implicit secrets**
   - Whenever a feature requires sensitive values (database URIs, tokens, paths), require them via `.env`.
   - If the value is unknown, add a placeholder in `.env.example` and a clear note in documentation.

7. **Minimal tests and manual verification**
   - Where tests already exist, keep them passing.
   - For new modules, add minimal smoke tests when feasible, focused on critical functionality.
   - Prioritize code clarity and explicit checks over sophisticated test frameworks.

8. **Logs and errors**
   - Prefer explicit exceptions with clear messages over silent failures.
   - Add structured logging calls in service and pipeline entry points.

9. **Reproducibility**
   - Ensure that experiments can be re-run using the same configuration and produce comparable results.
   - Use deterministic random seeds where relevant.

---

## 1. Project Goal and Scope

You are working on a backend service **scientific-api** that must:

- Build and manipulate graphs derived from:
  - **Cosmology**: large-scale structure from SDSS DR17 and DESI DR2 galaxy catalogs.
  - **Quantum mechanics**: discretized models of quantum systems represented as lattices and interaction graphs.
- Compute **graph features**:
  - Topological descriptors.
  - Spectral descriptors.
  - Node embeddings.
- Train and run **ML models**:
  - Classify graphs by origin (cosmological vs quantum).
  - Quantify similarity between graphs.
  - Cluster graphs in feature space.
- Compute **geometric similarity metrics** of graphs:
  - Gromov–Wasserstein and related distances.
  - Spectral and distribution-based distances.
- Expose functionality through **FastAPI**:
  - Experiment creation and configuration.
  - Launch of full pipeline.
  - Access to metrics, graphs and basic visualizations.

The implementation must be sufficient to support the thesis experiments and screenshots, not to operate as a production-grade SaaS.

---

## 2. Technology Stack

When you create or refactor code, conform to this stack:

- **Language**: Python 3.11
- **Web Framework**: FastAPI + Uvicorn
- **Scientific and graph libraries**:
  - `numpy`, `scipy`, `pandas`
  - `networkx` for graph representation
  - `scikit-learn` for classical ML
  - `pot` (Python Optimal Transport) for Gromov–Wasserstein distances
  - `matplotlib`, `plotly` for plotting and visualizations
- **Optional deep learning layer for GNN experiments**:
  - `torch`, `torch_geometric`
- **Data storage**:
  - MongoDB as primary document database
- **Caching and tasks**:
  - Redis for caching and Celery backend
  - Celery for asynchronous execution of heavy pipeline steps
- **Containerization**:
  - Docker and Docker Compose
- **Configuration**:
  - `pydantic.Settings` for application settings
  - `.env` with `.env.example` template
- **Code quality**:
  - Black for formatting
  - Ruff for linting

When adding or changing dependencies, update `pyproject.toml` and keep versions consistent.

---

## 3. Repository Structure

Target structure for the repository:

- `/app`
  - `/app/api` — FastAPI routers, request/response schemas, dependencies
  - `/app/core` — configuration, logging, shared utilities, error handling
  - `/app/services` — orchestration and business logic (pipeline coordination)
  - `/app/db` — MongoDB connection and models
- `/ml`
  - `/ml/data_cosmology` — ingestion, filtering and preprocessing for SDSS and DESI data
  - `/ml/data_quantum` — generation and processing of quantum models
  - `/ml/graphs` — graph construction and normalization
  - `/ml/features` — topological, spectral features and embeddings
  - `/ml/models` — training and inference routines
  - `/ml/metrics` — graph distance metrics and evaluation utilities
- `/configs` — experiment configuration files (YAML or JSON)
- `/data`
  - `/data/raw` — references to raw data locations, lightweight samples
  - `/data/processed` — generated graph and feature files
- `/notebooks` — research notebooks used in the thesis
- `/scripts` — helper scripts (data sampling, seeding, maintenance)
- `/tests` — minimal smoke tests
- Root files:
  - `README.md`
  - `TECHNICAL_PLAN.md`
  - `ROADMAP.md` (end-user oriented)
  - `docker-compose.yml`
  - `Dockerfile`
  - `pyproject.toml`
  - `.env.example`
  - `.vscode/` settings and extension recommendations
  - `scientific-api.http` for REST Client

Your task is to align current project files with this structure and create missing directories and modules when they are required by this plan.

---

## 4. Configuration and Environment

Tasks for you:

1. **Central configuration**
   - Create `/app/core/config.py` with a `Settings` class based on `pydantic.Settings`.
   - Settings must include:
     - `MONGO_URI`
     - `MONGO_DB_NAME`
     - `REDIS_URL`
     - `LOG_LEVEL`
     - `DATA_ROOT`
     - Optional ML-related toggles where necessary.
   - Provide a single `get_settings()` function that caches the loaded settings.

2. **Environment files**
   - Create `.env.example` with placeholder values for all settings used in `Settings`.
   - Ensure `TECHNICAL_PLAN.md` and `README.md` explain how to create `.env` from `.env.example`.

3. **Docker integration**
   - Ensure `Dockerfile` and `docker-compose.yml`:
     - Use Python 3.11.
     - Install requirements from `pyproject.toml`.
     - Expose port 8000.
     - Connect FastAPI container to MongoDB and Redis containers.

4. **Logging**
   - Add `/app/core/logging.py`:
     - Configure structured logging (standard library or `structlog`).
     - Provide `get_logger(name: str)` for consistent logger creation.
   - Use this logger in service and API layers instead of print statements.

---

## 5. Data Ingestion and Preprocessing

### 5.1 Cosmological Data

Tasks:

1. Create `/ml/data_cosmology/io.py`:
   - Functions to load raw galaxy catalogs or subsamples from SDSS DR17 and DESI DR2.
   - For this project, implement loading from local CSV or Parquet files located in `/data/raw/cosmology`.

2. Create `/ml/data_cosmology/preprocessing.py`:
   - Functions to:
     - Filter galaxies by RA, Dec, redshift ranges.
     - Compute 3D Cartesian coordinates from RA, Dec, redshift.
     - Save processed tables in `/data/processed/cosmology` in Parquet format.

3. Provide at least one configuration file in `/configs/cosmology_sample.yml` that defines:
   - Data source identifier.
   - RA, Dec, redshift ranges.
   - Any additional filters.

### 5.2 Quantum Data

Tasks:

1. Create `/ml/data_quantum/models.py`:
   - Implementation of parametrized quantum systems:
     - Two-dimensional harmonic oscillator.
     - Two-dimensional potential well with local perturbations.
   - Each model must have:
     - A function that returns potential values on a grid.
     - A function that constructs a discrete Hamiltonian matrix.

2. Create `/ml/data_quantum/preprocessing.py`:
   - Functions that:
     - Build a grid with given resolution.
     - Apply the chosen quantum model to compute the Hamiltonian.
     - Save resulting matrices and grid coordinates to `/data/processed/quantum`.

3. Add configuration files in `/configs/quantum` that define:
   - Grid resolution.
   - Model parameters.
   - Output paths.

---

## 6. Unified Graph Representation and Construction

### 6.1 Base Graph Model

Tasks:

1. Create `/ml/graphs/base.py`:
   - Define a `GraphObject` abstraction or a small set of utility functions over `networkx.Graph`.
   - Ensure that graphs contain:
     - Node IDs.
     - Normalized coordinates in a common space.
     - Node attributes (such as mass, luminosity, potential value).
     - Edge weights representing an effective distance or interaction strength.

2. Provide helper functions:
   - `normalize_coordinates(nodes)` to map positions into a unit cube.
   - `normalize_edge_weights(graph)` to scale weights into a fixed range.

### 6.2 Cosmology Graph Builder

Tasks:

1. Create `/ml/graphs/cosmology_builder.py`:
   - Function to build graphs from processed cosmology tables:
     - Use k-nearest neighbors in coordinate space with fixed `k` value.
     - Assign edge weights equal to normalized distances.
   - Save graphs in `/data/processed/cosmology/graphs`:
     - Use a consistent format (edge list plus node attributes table).

2. Ensure parameters like `k`, distance metric and filtering rules are driven by configuration files from `/configs`.

### 6.3 Quantum Graph Builder

Tasks:

1. Create `/ml/graphs/quantum_builder.py`:
   - Function to build graphs from Hamiltonian matrices and grid coordinates:
     - Nodes correspond to grid points.
     - Edges correspond to non-zero entries in the Hamiltonian.
     - Edge weights encode interaction strength.
   - Apply coordinate normalization to place points into the same unit cube as cosmological graphs.

2. Save graphs in `/data/processed/quantum/graphs` using the same format as the cosmology graphs.

### 6.4 Graph Consistency

Tasks:

1. Implement utilities in `/ml/graphs/utils.py`:
   - Functions that downsample or subsample larger graphs to a target number of nodes.
   - Functions that ensure consistent scaling of weights across both domains.

2. Ensure that configuration files specify target node counts and scaling behavior.

---

## 7. Feature Engineering

### 7.1 Topological Features

Tasks:

1. Create `/ml/features/topology.py`:
   - Function `compute_topology_features(graph)` that returns a fixed-size feature vector.
   - Include:
     - Degree distribution statistics.
     - Average clustering coefficient.
     - Average shortest path length for the largest connected component.
     - Effective diameter approximation.
     - Basic centrality statistics for a sample of nodes.

2. Ensure the output is convertible to a row in a Pandas DataFrame.

### 7.2 Spectral Features

Tasks:

1. Create `/ml/features/spectral.py`:
   - Function `compute_spectral_features(graph, k: int)`:
     - Build normalized Laplacian.
     - Compute the first `k` eigenvalues.
     - Produce summary statistics and a fixed-length vector.

2. Provide a default `k` in configuration and allow override via function arguments.

### 7.3 Embeddings

Tasks:

1. Create `/ml/features/embeddings.py`:
   - Use a node-level embedding method such as node2vec or a similar random-walk-based approach.
   - For each graph:
     - Compute node embeddings.
     - Aggregate them into:
       - Mean embedding.
       - Selected statistics over coordinates of embeddings.

2. Ensure output is compatible with the tabular feature representation.

### 7.4 Feature Table Construction

Tasks:

1. Create `/ml/features/feature_table.py`:
   - High-level functions that:
     - Load graphs from disk.
     - Compute topological, spectral and embedding features.
     - Save feature tables to `/data/processed/features` in Parquet format.
   - Ensure the table contains:
     - Graph ID.
     - System type (cosmology or quantum).
     - All feature columns.

---

## 8. Machine Learning Models

### 8.1 Classification

Tasks:

1. Create `/ml/models/classification.py`:
   - Pipeline to train models that distinguish cosmological and quantum graphs.
   - Implement:
     - Logistic Regression baseline.
     - Gradient boosting model (for example XGBoost or a similar library).
   - Provide functions:
     - `train_classifiers(feature_table)`.
     - `evaluate_classifiers(feature_table)`.

2. Store models in a dedicated directory, such as `/data/models/classification`.

### 8.2 Similarity Regression

Tasks:

1. Create `/ml/models/similarity_regression.py`:
   - Functions to:
     - Construct pairwise feature representations of graph pairs.
     - Train regression models that predict a similarity score between graphs based on:
       - Graph distance metrics.
       - Feature differences.

2. Save trained models in `/data/models/similarity`.

### 8.3 Clustering

Tasks:

1. Create `/ml/models/clustering.py`:
   - Functions to:
     - Apply k-means clustering to graphs in feature space.
     - Apply density-based clustering such as DBSCAN.
   - Provide utilities to label graphs with cluster IDs and compute cluster-level summaries.

---

## 9. Geometric Similarity Metrics

### 9.1 Gromov–Wasserstein

Tasks:

1. Create `/ml/metrics/gw_distance.py`:
   - Wrapper functions over POT library that:
     - Compute Gromov–Wasserstein distance between two graphs using distance matrices.
     - Optionally compute fused Gromov–Wasserstein when node features exist.
   - Ensure functions:
     - Accept graphs or precomputed distance matrices.
     - Return a scalar distance and any relevant diagnostics.

2. Provide high-level utilities to:
   - Compute distances between sets of graphs.
   - Store results in tables under `/data/processed/distances`.

### 9.2 Spectral and Distribution Distances

Tasks:

1. Create `/ml/metrics/spectral_distance.py`:
   - Function to compute L2 distance between sorted Laplacian spectra of two graphs.

2. Create `/ml/metrics/distribution_distance.py`:
   - Functions to compute distances between:
     - Degree distributions.
     - Community size distributions.
   - Use Wasserstein distance as appropriate.

3. Provide a consolidated function that takes two graphs and returns a vector of distance metrics.

---

## 10. Experiment Orchestration and Services

### 10.1 Experiment Entity

Tasks:

1. Create `/app/api/schemas/experiments.py`:
   - Pydantic models that define:
     - Experiment creation payload.
     - Experiment metadata response.
     - Experiment run status and results.

2. Create `/app/db/experiments.py`:
   - Functions to:
     - Insert new experiment documents into MongoDB.
     - Retrieve and update experiments.

### 10.2 Experiment Runner

Tasks:

1. Create `/app/services/experiment_runner.py`:
   - High-level orchestration of:
     - Data preparation.
     - Graph construction.
     - Feature computation.
     - Model training and evaluation.
     - Distance computation.
   - Expose a clear API:
     - `create_experiment(config)`
     - `run_full_pipeline(experiment_id, force_recompute: bool)`
     - `get_results(experiment_id)`

2. Integrate Celery where heavy steps are executed:
   - Define Celery tasks that wrap long-running computations.
   - Ensure idempotency where possible.

---

## 11. FastAPI Integration

Tasks:

1. Create `/app/api/routes/experiments.py`:
   - Implement endpoints under `/api/v1/experiments`:
     - `POST /experiments/` — create experiment.
     - `GET /experiments/{id}` — get experiment metadata.
     - `POST /experiments/{id}/run-full-pipeline` — trigger pipeline run.
     - `GET /experiments/{id}/results` — get aggregated metrics.
     - `GET /experiments/{id}/metrics` — get full set of metric values.
     - `GET /experiments/{id}/graphs/{system_type}` — provide graphs for visualization.
     - `GET /experiments/{id}/plots/{plot_type}` — provide plot data or images.

2. Register these routes in the main FastAPI application module.
3. Ensure OpenAPI schema is generated correctly and endpoints have clear request and response models.

---

## 12. Visualizations

Tasks:

1. Create `/app/services/visualization.py`:
   - Functions to prepare:
     - Degree distribution plots.
     - Spectral density plots.
     - Two-dimensional projections of embeddings.

2. Create notebooks in `/notebooks`:
   - `01_cosmology_graphs.ipynb`
   - `02_quantum_graphs.ipynb`
   - `03_spectral_analysis.ipynb`
   - `04_gw_distances.ipynb`
   - `05_ml_results.ipynb`
   - Each notebook must demonstrate one part of the pipeline and produce figures suitable for inclusion in the thesis.

3. Ensure API endpoints call the shared visualization functions instead of duplicating plotting logic.

---

## 13. Minimal Tests and Quality Checks

Tasks:

1. Under `/tests`, create smoke tests for:
   - Importability of main modules.
   - Basic FastAPI route responses (status codes and simple payloads).
   - At least one feature computation and one graph builder on a tiny synthetic dataset.

2. Configure pytest using existing `.vscode/settings.json` and `pyproject.toml`.

3. Ensure that `ruff` does not report severe errors for new code.

---

## 14. VS Code and Extension Usage

The repository includes `.vscode/settings.json` and `.vscode/extensions.json`.  
You must generate code compatible with these tools:

1. **GitHub Copilot and Copilot Chat**
   - When writing code, follow the patterns defined in this plan so that suggestions remain consistent.
   - Use this `TECHNICAL_PLAN.md` as the main context when responding to user prompts in Copilot Chat.

2. **Python and Pylance**
   - Keep type hints consistent to maintain high-quality editor support.
   - Avoid dynamic patterns that confuse static analysis without necessity.

3. **Jupyter**
   - When creating notebooks, keep imports and paths aligned with project structure so relative imports work from repo root.

4. **Ruff and Black**
   - Write code that is already close to Black formatting and Ruff linting to minimize automatic changes.
   - Keep line length close to 88 characters.

5. **REST Client**
   - Ensure endpoints match what is defined in `scientific-api.http`.
   - Maintain backward compatibility of request and response shapes as the project evolves.

---

## 15. Checklist

Use this checklist when working on the project.  
Mark items as completed when the corresponding tasks are fully implemented and tested.

### Infrastructure

- [x] Central configuration in `/app/core/config.py` implemented.
- [x] Logging configured in `/app/core/logging.py`.
- [x] `.env.example` created and used.
- [x] Dockerfile and `docker-compose.yml` aligned with Python 3.11 and services.

### Data Layer

- [x] Cosmology ingestion and preprocessing modules implemented.
- [x] Quantum model generation and preprocessing modules implemented.
- [ ] Processed data written to `/data/processed`.

### Graphs

- [x] Base graph utilities created.
- [x] Cosmology graph builder implemented and tested on sample data.
- [x] Quantum graph builder implemented and tested on sample data.
- [x] Consistency utilities for node and weight normalization implemented.

### Features

- [ ] Topological feature computation implemented.
- [ ] Spectral feature computation implemented.
- [ ] Embeddings computation and aggregation implemented.
- [ ] Unified feature table construction implemented.

### Models

- [ ] Classification models implemented and evaluated.
- [ ] Similarity regression models implemented.
- [ ] Clustering routines implemented.

### Metrics

- [ ] Gromov–Wasserstein distance computation implemented.
- [ ] Spectral distance computation implemented.
- [ ] Distribution distance computation implemented.
- [ ] Consolidated distance interface implemented.

### Orchestration and API

- [ ] Experiment schemas and DB layer implemented.
- [ ] Experiment runner service implemented.
- [ ] FastAPI routes for experiments and metrics implemented.
- [ ] Plots and graph endpoints implemented.

### Visualizations and Notebooks

- [ ] Core notebooks created and runnable.
- [ ] Plots suitable for thesis included.

### Quality

- [ ] Minimal smoke tests created and passing.
- [ ] Ruff and Black applied to new code.
- [ ] Documentation updated where behavior changed.

---

When you, as an AI assistant, receive a prompt from the user about this project, you must:

1. Identify relevant sections of this plan.
2. Work only on the described tasks.
3. Produce clean, minimal code that adheres to this structure.
4. Update this plan only when required to reflect the actual state of the repository.