from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class VectorStore(ABC):
    @abstractmethod
    def add_embeddings(self, embeddings: List[List[float]]):
        """
        Add face embeddings.

        Args:
            embeddings (List[List[float]]): A list of face embeddings to add.
        """

    @abstractmethod
    def search_with_score(
        self,
        embedding: List[float],
        k: int = 1,
    ) -> Tuple[np.ndarray[float], np.ndarray[int]]:
        """
        Search for the nearest embeddings to the face embedding.

        Args:
            embedding (List[float]): The input embedding to search for.
            k (int): The number of nearest embeddings to retrieve (default: 1).

        Returns:
            A tuple containing the scores and indices of the nearest embeddings.
        """
