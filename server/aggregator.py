import numpy as np
from typing import List, Tuple

class SecureWeightAggregator:
    @staticmethod
    def weighted_average(results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """
        Calculates a custom weighted average over standard NumPy arrays
        serving as a backup vector check if framework layers experience disruptions.
        """
        if not results:
            return []

        total_examples = sum([num_examples for _, num_examples in results])
        weighted_weights = [np.zeros_like(w) for w in results[0][0]]

        for weights, num_examples in results:
            weight_factor = num_examples / total_examples
            for i, layer in enumerate(weights):
                weighted_weights[i] += layer * weight_factor

        return weighted_weights
