INSTRUCTION FOR gpt-5.1-codex-max (VS Code, repo: scientific-api)
0) Non-negotiables
No “optional”, no alternatives. Implement exactly as written.
Idempotent changes. If a file/dir exists:
do not delete;
do not overwrite blindly;
only append/patch with minimal diffs;
keep backward compatibility when possible.
Reproducible research. Fix seeds; persist configs; persist artifacts; persist dataset manifests.
Large training data requirement. Do not rely on a single 50k-point set. Generate hundreds/thousands of graph samples from the same catalogs via spatial windowing / bootstrapped subvolumes to build a large graph-level dataset for ML.
Every code cell in notebooks must have: a Markdown explanation before it, and a Markdown “What we learned / checks passed” after it.
Raw column names may differ. Implement robust column mapping with explicit candidate lists and hard failures with diagnostics.
1) Repo state handling (when files already exist)
1.1 Create missing dirs only
Use Path(...).mkdir(parents=True, exist_ok=True) everywhere. Never assume empty tree.
1.2 Patch existing files safely
For .gitignore: append missing patterns only if not present.
For .devcontainer/devcontainer.json: patch keys without removing existing user settings. Preserve formatting.
For python modules: if a module exists, update functions in-place; do not duplicate.
1.3 Add a single top-level research entrypoint
Create or update:
scientific_api/pipeline/cosmology_ingest.py
scientific_api/pipeline/graph_dataset.py
scientific_api/pipeline/features.py
They must be importable and callable from notebooks.
2) Directory layout (must exist after implementation)
Ensure these exist (create if missing):
configs/
notebooks/
data/raw/desi/dr1/
data/raw/sdss/dr17/
data/processed/cosmology/low_z/
data/processed/cosmology/high_z/
data/processed/cosmology/manifests/
outputs/pipeline/low_z/
outputs/pipeline/high_z/
outputs/datasets/
reports/figures/
reports/tables/
​
Update .gitignore to include (add only if absent):
data/
outputs/
reports/
*.fits
*.fits.gz
*.parquet
*.arrow
​
3) Fixed presets (do not change values)
Create/patch configs/cosmology_presets.yaml with exactly:
presets:
  low_z:
    region:
      ra_min: 150.0
      ra_max: 210.0
      dec_min: -5.0
      dec_max: 5.0
    redshift:
      z_min: 0.02
      z_max: 0.30
    limits:
      target_points_per_source: 500000
      seed: 42
    sdss_dr17:
      class: "GALAXY"
      zwarning_allowed: [0, 16]
      ra_bin_deg: 5.0
      per_query_limit: 100000
      max_queries: 200
    desi_dr1:
      files:
        - { target: "BGS_BRIGHT", photsys: "N" }
        - { target: "BGS_BRIGHT", photsys: "S" }

  high_z:
    region:
      ra_min: 150.0
      ra_max: 210.0
      dec_min: -5.0
      dec_max: 5.0
    redshift:
      z_min: 0.45
      z_max: 1.05
    limits:
      target_points_per_source: 500000
      seed: 42
    sdss_dr17:
      class: "GALAXY"
      zwarning_allowed: [0, 16]
      ra_bin_deg: 5.0
      per_query_limit: 100000
      max_queries: 200
    desi_dr1:
      files:
        - { target: "LRG", photsys: "N" }
        - { target: "LRG", photsys: "S" }
​
Key requirement: 500k points per source target (SDSS and DESI each). If SDSS cannot reach it (likely for high_z), ingestion must:
collect as many as possible within limits,
then downsample DESI to match SDSS for fair comparisons,
and record actual counts in manifest.
No silent behavior.
4) Devcontainer: must support FITS + Parquet robustly
If .devcontainer/devcontainer.json exists, patch it; otherwise create.
Target base image (must be Debian/Ubuntu-based):
mcr.microsoft.com/devcontainers/python:3.11-bookworm
Ensure postCreateCommand (merge, do not replace) installs:
pip install -U pip
pip install -e .
pip install numpy pandas scipy scikit-learn matplotlib pyarrow pyyaml httpx astropy tqdm networkx
If repo already uses requirements/pyproject, also add these deps there (minimal, deterministic).
5) Source acquisition implementation (SDSS + DESI)
5.1 SDSS DR17 via SkyServer SqlSearch (CSV)
Create scientific_api/data_sources/cosmology/sdss_dr17_sql.py with:
Constant
SDSS_SQL_URL = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"
​
Core function
def fetch_sdss_dr17_points_rect_tiled(preset: dict, preset_name: str, out_raw_dir: Path) -> Path:
    """
    Returns path to a single merged CSV (raw) for this preset.
    Must tile RA into bins to exceed single-query limits and build large dataset.
    """
​
Hard algorithm
Read ra_min, ra_max, dec_min, dec_max, z_min, z_max.
Tile RA into bins of width ra_bin_deg (default 5°).
For each tile [ra_i, ra_j] run a query with TOP per_query_limit.
Concatenate results; drop duplicates by obj_id.
Stop when either:
unique count >= target_points_per_source, or
queries == max_queries.
Save merged raw CSV to:
data/raw/sdss/dr17/{preset_name}/sdss_dr17__{preset_name}__raw.csv
Write manifest JSON next to it with:
query count, tile list, row counts before/after dedup, timestamp, git commit.
SQL text template (use aliases to force stable column names)
Use PhotoObj + SpecObj join (stable for ra/dec + z/class/zwarning). Always alias columns to:
obj_id, ra_deg, dec_deg, z, zwarning, class, mag_r
Example per tile (values substituted):
SELECT TOP {LIMIT}
  p.objid        AS obj_id,
  p.ra           AS ra_deg,
  p.dec          AS dec_deg,
  s.z            AS z,
  s.zWarning     AS zwarning,
  s.class        AS class,
  p.psfMag_r     AS mag_r
FROM PhotoObj AS p
JOIN SpecObj  AS s ON s.bestObjID = p.objID
WHERE
  p.ra  BETWEEN {RA_MIN} AND {RA_MAX}
  AND p.dec BETWEEN {DEC_MIN} AND {DEC_MAX}
  AND s.z BETWEEN {Z_MIN} AND {Z_MAX}
  AND s.class = '{CLASS}'
  AND s.zWarning IN ({ZWARN_LIST})
ORDER BY p.objid
​
Networking
Use httpx.Client(timeout=60); retries = 3; backoff = 2, 4, 8 sec.
Fail hard if CSV parse returns zero columns or missing required columns.
5.2 DESI DR1 clustering FITS
Create scientific_api/data_sources/cosmology/desi_dr1_lss.py with:
Constants
DESI_DR1_CLUSTERING_BASE = "https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/clustering/"
FILENAME_TEMPLATE = "{TARGET}_{PHOTSYS}_clustering.dat.fits"
​
Functions
download_desi_dr1_clustering(target, photsys, out_dir) -> Path
load_desi_clustering_points(fits_path) -> pandas.DataFrame
Robust raw column mapping (MANDATORY)
Create scientific_api/data_processing/cosmology/column_map.py with:
DESI_COLUMN_CANDIDATES = {
  "ra_deg":  ["RA", "ra", "TARGET_RA", "target_ra"],
  "dec_deg": ["DEC", "dec", "TARGET_DEC", "target_dec"],
  "z":       ["Z", "z", "Z_COSMO", "z_cosmo", "Z_NOT4CLUS", "z_not4clus"],
  "weight":  ["WEIGHT", "weight", "WEIGHT_FKP", "weight_fkp", "WEIGHT_TOT", "weight_tot"]
}
​
Selection rule:
pick the first existing candidate for each required field;
if any required field missing → raise ValueError with:
list of available columns (first 200),
which fields failed,
file path.
This directly addresses “raw columns may have other names”.
6) Normalization contract (single schema) + comoving coordinates
Create scientific_api/data_processing/cosmology/schema.py:
Required final schema (Parquet)
source (str) : "sdss_dr17" or "desi_dr1"
sample (str) : e.g. "GALAXY_low_z", "BGS_BRIGHT_N_low_z", "LRG_S_high_z"
obj_id (str)
ra_deg (float64)
dec_deg (float64)
z (float64)
x_mpc, y_mpc, z_mpc (float64)
weight (float64)
Create scientific_api/data_processing/cosmology/coords.py:
Use astropy.cosmology.Planck18
Convert RA/DEC/Z → comoving (x,y,z) in Mpc
Must be vectorized (numpy arrays), not per-row python loop.
Create scientific_api/data_processing/cosmology/normalize_points.py:
normalize_sdss_csv_to_parquet(raw_csv_path, preset, preset_name) -> Path
normalize_desi_fits_to_parquet(fits_paths, preset, preset_name) -> Path
Apply identical cuts: RA/DEC/Z rectangle.
Clean rules (hard):
drop NaNs in required cols
enforce finite floats
enforce z > 0
enforce weight > 0 (SDSS weight = 1.0)
Save:
data/processed/cosmology/{preset_name}/sdss_dr17.parquet
data/processed/cosmology/{preset_name}/desi_dr1.parquet
Also write dataset manifest:
data/processed/cosmology/manifests/{preset_name}.json
with counts before/after cuts, and final matched counts used for comparisons.
Fairness rule (hard)
After normalization:
compute n_sdss, n_desi
define n = min(n_sdss, n_desi)
downsample both to n using fixed seed 42 and persist the sampled parquet files:
sdss_dr17__matched.parquet
desi_dr1__matched.parquet
7) Ingestion entrypoint (single command)
Create scientific_api/pipeline/cosmology_ingest.py:
CLI:
python -m scientific_api.pipeline.cosmology_ingest --preset low_z
python -m scientific_api.pipeline.cosmology_ingest --preset high_z
​
Behavior (hard order):
Load preset from YAML.
SDSS tiled SQL → raw merged CSV.
DESI download required FITS files.
Normalize SDSS + DESI to Parquet + comoving coords.
Create matched Parquets (same n).
Write:
outputs/ingestion/{preset}/run_meta.json (commit, timestamp, counts, paths)
Must print a short deterministic summary at end:
preset, n_sdss_raw, n_desi_raw, n_sdss_matched, n_desi_matched, output paths.
8) “Many data” for ML: generate a large graph-level dataset
This is mandatory to satisfy “serious training”.
Create scientific_api/pipeline/graph_dataset.py:
8.1 Graph sampling strategy (hard)
Input: matched Parquet points for SDSS and DESI.
Create many graphs by spatial windowing in comoving space:
Define a cubic window size L = 200 Mpc
Sample N_graphs_per_source_per_preset = 500 (total graphs = 2 sources × 2 presets × 500 = 2000 graphs)
For each graph:
pick a random center point among available points,
take all points within [x±L/2, y±L/2, z±L/2],
if window has < 800 nodes → resample center,
if window has > 4000 nodes → downsample to 4000 with seed derived from graph id,
build kNN graph with k=12 in 3D for those nodes.
Persist each graph’s node table and edge list:
outputs/datasets/graphs/{preset}/{source}/graph_{idx:04d}_nodes.parquet
outputs/datasets/graphs/{preset}/{source}/graph_{idx:04d}_edges.parquet
Also persist a registry CSV:
outputs/datasets/graph_registry.csv
columns:
preset, source, graph_id, n_nodes, n_edges, center_x, center_y, center_z, path_nodes, path_edges
8.2 Feature extraction for all graphs (hard)
Create scientific_api/pipeline/features.py:
For each graph compute and store one row:
graph size: n_nodes, n_edges, density
degree stats: mean, std, skew (if easy), max
clustering: mean
components: n_components, giant_frac
shortest-path proxy: approximate via sampling 200 nodes (avoid O(n^2))
spectral: first 30 eigenvalues of normalized Laplacian (use sparse methods; if fails, fallback to smallest 30 of dense for n<=2000; must be deterministic)
spectral summary: mean, std, spectral_gap = λ2−λ1 (λ1 ~ 0)
Persist:
reports/tables/graph_features.parquet
reports/tables/graph_features.csv (small enough)
8.3 ML tasks (hard)
In Notebook 02 (see below), train and evaluate:
Source classifier: predict source (SDSS vs DESI) from graph features.
Preset classifier: predict preset (low_z vs high_z) from graph features.
Report:
accuracy, F1, ROC-AUC (for binary tasks),
confusion matrix,
feature importance (tree-based) or coefficients (logreg).
Use:
baseline: LogisticRegression (standardized)
strong: RandomForestClassifier (fixed hyperparams, deterministic)
9) Notebook requirements: two notebooks, full explanations per cell
Notebook 01:
notebooks/01_pipeline_implementation.ipynb
Purpose: implement + validate ingestion + normalization + coordinate transform + sanity plots.
Cell-by-cell structure (must follow):
[M1] Title & scope (Markdown)
What this notebook produces (files, figures, tables).
Exact presets used.
[C1] Reproducibility bootstrap (Code)
set seeds (random/numpy)
print python version
print git rev-parse HEAD
define ROOT, DATA_DIR, OUTPUT_DIR
[M2] Config load explanation
describe preset YAML and why fixed.
[C2] Load presets (Code)
load YAML
validate keys exist (raise if missing)
[M3] Ingestion execution explanation
states: SDSS tiled SQL, DESI FITS download, normalize, matched parquet.
[C3] Run ingestion for low_z (Code)
call module function, not shell strings
assert output files exist
[C4] Run ingestion for high_z (Code)
same
[M4] Raw/processed data sanity checks (Markdown)
[C5] Load matched parquets (Code)
load SDSS+DESI matched for low_z
print counts, min/max z, RA/DEC bounds
[C6] Plot 1: RA/DEC scatter (Code)
Save: outputs/pipeline/low_z/ra_dec_sdss.png, .../ra_dec_desi.png
Check: visually overlap in region
[C7] Plot 2: z histogram overlay (Code)
Save: outputs/pipeline/low_z/z_hist_overlay.png
Check: comparable distributions (report if not)
[C8] Plot 3: comoving x-y scatter (Code)
Save: outputs/pipeline/low_z/xy_overlay.png
Repeat C5–C8 for high_z, saving into outputs/pipeline/high_z/….
[M5] Column mapping validation (Markdown)
explicitly state: raw names may vary; show mapping chosen.
[C9] Print DESI mapping diagnostics (Code)
show which raw columns were used for ra/dec/z/weight.
[M6] Pipeline completion criteria (Markdown)
[C10] Final checklist table (Code)
make a small dataframe: counts, file paths, sha256 presence, manifests present
display it.
Notebook 02:
notebooks/02_experiments_and_results.ipynb
Purpose: build large graph dataset, compute features, train ML, analyze stability, produce publication-grade figures/tables.
Cell-by-cell structure:
[M1] Research questions & hypotheses
H1: graph geometry differs by source (SDSS vs DESI)
H2: graph geometry differs by z-regime (low vs high)
H3: stability vs k (kNN)
[C1] Reproducibility bootstrap
seeds, versions, git commit, dirs
[M2] Load processed data
must use matched parquets only.
[C2] Load matched parquets for both presets
counts table
[M3] Build large graph dataset plan
describe windowing in comoving space, n_graphs, L=200 Mpc.
[C3] Generate graph dataset
call graph_dataset.build_graph_corpus(...)
assert registry exists and has expected row count (~2000)
[C4] Plot: node count distribution
histogram of n_nodes by source/preset
Save: reports/figures/graph_nodecount_dist.png
[M4] Feature extraction
list all computed features including Laplacian spectrum.
[C5] Compute features for all graphs
persist reports/tables/graph_features.parquet/csv
assert no NaNs in required feature columns
[C6] Plot: feature correlation heatmap
Save: reports/figures/feature_corr.png
[M5] ML Experiment 1: source classification
[C7] Train/test split (Code)
stratified split by source, fixed random_state
[C8] Train LogisticRegression baseline
report metrics table
Save: reports/tables/ml_source_logreg_metrics.csv
[C9] Train RandomForest
report metrics table
Save: reports/tables/ml_source_rf_metrics.csv
[C10] Confusion matrix plot
Save: reports/figures/source_confusion_rf.png
[C11] Feature importance plot (RF)
Save: reports/figures/source_feature_importance_rf.png
[M6] ML Experiment 2: preset classification (low_z vs high_z)
Repeat analogous cells:
Save: reports/figures/preset_confusion_rf.png
Save metrics in reports/tables/…
[M7] Sensitivity analysis vs k
k grid fixed: {8, 12, 16}
[C12] Rebuild features for k=8,12,16 (Code)
do not rebuild whole corpus; reuse node windows, only rebuild edges for new k
persist reports/tables/sensitivity_k.parquet
[C13] Plot: key metrics vs k
choose 6 key metrics (avg_degree, clustering_mean, giant_frac, spectral_gap, eigen_mean, eigen_std)
Save: reports/figures/sensitivity_vs_k.png
[M8] Final summary (Markdown)
bullet conclusions, limitations, next steps
must reference produced tables/figures paths
[C14] Auto-generate reports/summary.md
write a short markdown file containing:
dataset sizes
best metrics
links (paths) to figures/tables
10) Plot specification (exact filenames + meaning)
All plots must be saved with these exact names (create directories if needed):
Pipeline (Notebook 01)
outputs/pipeline/low_z/ra_dec_sdss.png — RA vs Dec scatter (SDSS)
outputs/pipeline/low_z/ra_dec_desi.png — RA vs Dec scatter (DESI)
outputs/pipeline/low_z/z_hist_overlay.png — z hist overlay (SDSS vs DESI)
outputs/pipeline/low_z/xy_overlay.png — comoving x-y overlay (SDSS vs DESI)
Same four for high_z.
Research (Notebook 02)
reports/figures/graph_nodecount_dist.png
reports/figures/feature_corr.png
reports/figures/source_confusion_rf.png
reports/figures/source_feature_importance_rf.png
reports/figures/preset_confusion_rf.png
reports/figures/sensitivity_vs_k.png
11) Acceptance criteria (no ambiguity)
After implementation, the following must succeed in the container terminal:
python -m scientific_api.pipeline.cosmology_ingest --preset low_z
python -m scientific_api.pipeline.cosmology_ingest --preset high_z
​
These files must exist:
data/processed/cosmology/low_z/sdss_dr17__matched.parquet
data/processed/cosmology/low_z/desi_dr1__matched.parquet
data/processed/cosmology/high_z/sdss_dr17__matched.parquet
data/processed/cosmology/high_z/desi_dr1__matched.parquet
data/processed/cosmology/manifests/low_z.json
data/processed/cosmology/manifests/high_z.json
Notebook “Run All” must complete without errors:
notebooks/01_pipeline_implementation.ipynb
notebooks/02_experiments_and_results.ipynb
These outputs must exist after notebooks:
reports/tables/graph_features.parquet
reports/tables/graph_features.csv
reports/tables/sensitivity_k.parquet
reports/summary.md
All required figure PNGs listed above.
12) Implementation note about raw column names (must be enforced)
SDSS: you force stable names via SQL aliases (AS ra_deg, etc.). If CSV lacks them → fail with diagnostics (print header).
DESI: you must select columns via candidate mapping (Section 5.2) and print the resolved mapping into:
outputs/ingestion/{preset}/desi_column_mapping.json
No silent assumptions.