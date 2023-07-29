from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pinecone

from openfaceid.vector_stores import vector_store


class Pinecone(vector_store.VectorStore):
    def __init__(
        self,
        index: pinecone.Index,
    ):
        """
        Initialize the Pinecone Vector Store.

        Args:
            index: The Pinecone index to store embeddings.
        """

        self.index = index

    def add_embeddings(
        self,
        embeddings: List[Tuple[str, np.ndarray]],
    ) -> None:
        vector_embeddings = [
            (image_id, embedding.tolist()) for image_id, embedding in embeddings
        ]

        self.index.upsert(vector_embeddings)

    def search_with_score(
        self,
        embedding: np.ndarray,
        k: int = 1,
    ) -> Tuple[List[float], List[str]]:
        response = self.index.query(vector=embedding.tolist(), top_k=k)

        scores = [match["score"] for match in response.matches]
        image_ids = [match["id"] for match in response.matches]

        return scores, image_ids
