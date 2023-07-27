from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class VectorStore(ABC):
    @abstractmethod
    def add_embeddings(self, embeddings: List[Tuple[str, np.ndarray]]) -> None:
        """
        Add face embeddings.

        Args:
            embeddings: A list of face embeddings to add.
        """

    @abstractmethod
    def search_with_score(
        self,
        embedding: np.ndarray,
        k: int = 1,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Search for the nearest embeddings to the face embedding.

        Args:
            embedding: The input embedding to search for.
            k: The number of nearest embeddings to retrieve (default: 1).

        Returns:
            A tuple containing the scores and indices of the nearest embeddings.
        """
