import numpy as np
import pinecone
import pytest

from openfaceid.vector_stores.pinecone import Pinecone
from tests import helpers


@pytest.fixture
def pinecone_index() -> pinecone.Index:
    api_key = "d569c718-dae9-4a7b-b16c-b704e9dcd9fc"
    pinecone.init(api_key=api_key, environment="us-west4-gcp-free")
    # pinecone.create_index("open-face-id-tests", dimension=3, metric="euclidean")
    index = pinecone.Index("open-face-id-tests")

    return index


@pytest.mark.vcr()
def test_add_and_search_embeddings(pinecone_index: pinecone.Index) -> None:
    vector_store = Pinecone(pinecone_index)

    embeddings = [
        ("image1", np.array([0.1, 0.2, 0.3])),
        ("image2", np.array([0.4, 0.5, 0.6])),
    ]
    vector_store.add_embeddings(embeddings)

    embedding = np.array([0.2, 0.3, 0.4])
    scores, image_ids = vector_store.search_with_score(embedding)

    assert len(scores) == 1
    assert len(image_ids) == 1
    assert helpers.is_float_list(scores)
    assert helpers.is_str_list(image_ids)
