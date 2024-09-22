import numpy as np
from scipy.sparse import csr_matrix
from sympy import primerange
from numpy import log

class PaletteWL:
    def __init__(self, A, labels=None):
        """
        Initialize the PaletteWL class.
        
        Args:
            A (ndarray): Original adjacency matrix of the enclosing subgraph.
            labels (ndarray, optional): Initial labels (colors) of the nodes. Defaults to None.
        """
        self.A = csr_matrix(A)  # Use a sparse matrix for efficiency
        self.labels = labels if labels is not None else np.ones(A.shape[0], dtype=int)
        self.equivalence_classes = np.zeros_like(self.labels)

    def palette_wl(self):
        """
        Perform the Weisfeiler-Lehman color refinement until stability.
        
        Returns:
            ndarray: Final labels (equivalence classes).
        """
        while not np.array_equal(self.labels, self.equivalence_classes):
            self.equivalence_classes = self.labels.copy()
            self.labels = self.wl_transformation(self.A, self.labels)

        return self.equivalence_classes

    def wl_transformation(self, A, labels):
        """
        Apply the WL transformation to update labels.
        
        Args:
            A (csr_matrix): The adjacency matrix.
            labels (ndarray): The current labels (colors) of the nodes.
        
        Returns:
            ndarray: Updated labels after the WL transformation.
        """
        num_labels = np.max(labels)

        # Generate prime numbers needed for the labels
        primes_required = np.array([2, 3, 7, 19, 53, 131, 311, 719, 1619, 3671, 
                                    8161, 17863, 38873, 84017, 180503, 386093, 
                                    821641, 1742537, 3681131, 7754077, 16290047])

        primes_needed = primes_required[int(np.ceil(np.log2(num_labels)))]

        # Use sympy to generate prime numbers (alternative to MATLAB's `primes`)
        primes = np.array(list(primerange(0, primes_needed + 1)))
        log_primes = np.log(primes[:num_labels])

        # Normalize the label signatures
        Z = np.ceil(np.sum(log_primes[labels - 1]))

        # Calculate the new signatures (labels + adjacency * log_primes)
        signatures = labels + A.dot(log_primes[labels - 1]) / Z

        # Map signatures to unique integers (equivalence classes)
        _, new_labels = np.unique(np.round(signatures, decimals=5), return_inverse=True)

        return new_labels + 1  # MATLAB indexing starts at 1, Python starts at 0

