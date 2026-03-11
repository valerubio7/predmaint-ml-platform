import pytest

from features.build_features import build_features
from ingestion.ingest import load_raw_data


@pytest.fixture
def raw_data():
    """Load raw dataset for testing."""
    return load_raw_data()


@pytest.fixture
def features(raw_data):
    """Build features from raw data."""
    X, y = build_features(raw_data)
    return X, y


def test_raw_data_shape(raw_data):
    """Dataset must have 10000 rows and 14 columns."""
    assert raw_data.shape == (10000, 14)


def test_no_missing_values(raw_data):
    """Raw dataset must have no missing values."""
    assert raw_data.isnull().sum().sum() == 0


def test_features_shape(features):
    """Feature matrix must have 10000 rows and 8 columns."""
    X, y = features
    assert X.shape == (10000, 8)


def test_target_is_binary(features):
    """Target must only contain 0 and 1."""
    _, y = features
    assert set(y.unique()) == {0, 1}


def test_no_missing_features(features):
    """Feature matrix must have no missing values after transformation."""
    X, _ = features
    assert X.isnull().sum().sum() == 0


def test_one_hot_encoding(features):
    """Type column must be one-hot encoded into three columns."""
    X, _ = features
    assert "Type_H" in X.columns
    assert "Type_L" in X.columns
    assert "Type_M" in X.columns
    assert "Type" not in X.columns
