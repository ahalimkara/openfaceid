import numpy as np
import pytest
from faiss import IndexFlatL2

from openfaceid.vector_stores.faiss import FAISS

pytestmark = pytest.mark.filterwarnings(
    "ignore:distutils Version classes are deprecated."
)


@pytest.fixture
def faiss_index():
    """Create a dummy FAISS index for testing."""
    dimension = 3
    return IndexFlatL2(dimension)


def test_faiss_add_embeddings(faiss_index):
    """Test adding vector embeddings with image ids."""
    faiss_store = FAISS(faiss_index)

    embeddings = [
        ("image1", [1.1, 2.2, 3.3]),
        ("image2", [4.4, 5.5, 6.6]),
    ]

    faiss_store.add_embeddings(embeddings)

    assert faiss_store.index.ntotal == len(embeddings)


def test_faiss_search_with_score(faiss_index):
    """Test vector embeddings search."""
    faiss_store = FAISS(faiss_index)

    embeddings = [
        ("1", [1.1, 2.2, 3.3]),
        ("2", [4.4, 5.5, 6.6]),
    ]
    faiss_store.add_embeddings(embeddings)

    test_embedding = [1.0, 2.0, 3.0]
    scores, image_ids = faiss_store.search_with_score(test_embedding)

    assert len(scores) == 1
    assert len(image_ids) == 1
    assert isinstance(image_ids[0], str)
    assert isinstance(scores[0], np.float32)


def test_faiss_add_embeddings_in_steps(faiss_index):
    """Test adding embeddings in multiple steps to check returning correct image id."""
    faiss_store = FAISS(faiss_index)

    embeddings = [
        ("1", [1.1, 2.2, 3.3]),
        ("2", [4.4, 5.5, 6.6]),
        ("3", [7.7, 8.8, 9.9]),
        ("4", [10.0, 11.1, 12.2]),
        ("5", [13.3, 14.4, 15.5]),
        ("6", [16.6, 17.7, 18.8]),
    ]

    faiss_store.add_embeddings(embeddings[:2])
    faiss_store.add_embeddings(embeddings[2:4])
    faiss_store.add_embeddings(embeddings[4:])

    test_embedding = [7.0, 8.0, 9.0]
    scores, image_ids = faiss_store.search_with_score(test_embedding)

    assert faiss_store.index.ntotal == len(embeddings)
    assert image_ids[0] == embeddings[2][0]


def test_faiss_search_with_score_multiple_results(faiss_index):
    """Test searching embeddings with more than one nearest neighbours."""
    faiss_store = FAISS(faiss_index)

    embeddings = [
        ("1", [1.1, 2.2, 3.3]),
        ("2", [4.4, 5.5, 6.6]),
        ("3", [7.7, 8.8, 9.9]),
    ]
    faiss_store.add_embeddings(embeddings)

    k = 2
    test_embedding = [1.0, 2.0, 3.0]
    scores, image_ids = faiss_store.search_with_score(test_embedding, k)

    assert len(scores) == 2
    assert len(image_ids) == 2
