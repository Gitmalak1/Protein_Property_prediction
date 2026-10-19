import pytest
from src.data_utils import extract_sequence_features

def test_composition_sums_to_one():
    features = extract_sequence_features("ACDEFGHIKLMNPQRSTVWY")
    assert sum(features[aa] for aa in "ACDEFGHIKLMNPQRSTVWY") == pytest.approx(1.0)
    assert features["sequence_length"] == 20

def test_invalid_sequence_rejected():
    with pytest.raises(ValueError):
        extract_sequence_features("ABZ")

def test_feature_values():
    features = extract_sequence_features("GGPP")
    assert features["glycine_fraction"] == pytest.approx(0.5)
    assert features["proline_fraction"] == pytest.approx(0.5)
