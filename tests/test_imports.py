import sys
from pathlib import Path
import pytest

def test_imports():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    pytest.importorskip("astropy")
    pytest.importorskip("scipy")

    import scientific_api
    from scientific_api.pipeline import cosmology_ingest
    from scientific_api.graphs.knn import GraphData
    from scientific_api.quantum.ensembles import quantum_graph_ensemble
    from scientific_api.features.graph_features import featurize_graph

    assert GraphData is not None
    assert quantum_graph_ensemble is not None
    assert featurize_graph is not None
