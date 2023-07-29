from __future__ import annotations

from typing import List, Tuple

import numpy as np
from faiss import IndexFlatL2

from openfaceid.vector_stores import vector_store


class FAISS(vector_store.VectorStore):
    def __init__(
        self,
        index: IndexFlatL2,
    ):
        """
        Initialize the FAISS Vector Store.

        Args:
            index: The FAISS index to store embeddings.
        """

        self.index = index
        self.index_to_image_id: dict[int, str] = {}

    def add_embeddings(
        self,
        embeddings: List[Tuple[str, np.ndarray]],
    ) -> None:
        image_ids, image_embeddings = zip(*embeddings)
        vector_embeddings = np.array(image_embeddings, dtype=np.float32)

        start_index = len(self.index_to_image_id)
        for index, image_id in enumerate(image_ids):
            self.index_to_image_id[start_index + index] = image_id

        self.index.add(vector_embeddings)

    def search_with_score(
        self,
        embedding: np.ndarray,
        k: int = 1,
    ) -> Tuple[List[float], List[str]]:
        vector_embeddings = np.array([embedding], dtype=np.float32)

        scores, indices = self.index.search(vector_embeddings, k)

        image_ids = []
        for index in indices[0]:
            if index in self.index_to_image_id:
                image_ids.append(self.index_to_image_id[index])

        return scores[0].tolist(), image_ids
