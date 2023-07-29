from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class VectorStore(ABC):
    @abstractmethod
    def add_embeddings(self, embeddings: List[Tuple[str, np.ndarray]]) -> None:
        """
        Add embeddings to the vector store.

        Args:
            embeddings: A list of tuples containing image IDs and
                corresponding face embeddings to add.
        """

    @abstractmethod
    def search_with_score(
        self,
        embedding: np.ndarray,
        k: int = 1,
    ) -> Tuple[List[float], List[str]]:
        """
        Search for the nearest embeddings to the face embedding.

        Args:
            embedding: The face embedding to search for.
            k: The number of nearest embeddings to retrieve (default: 1).

        Returns:
            A tuple containing the scores and image IDs of the nearest embeddings.
        """
