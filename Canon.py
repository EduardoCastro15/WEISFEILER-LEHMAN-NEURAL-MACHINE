import numpy as np
import networkx as nx

class Canon:
    def __init__(self, subgraph, classes=None):
        """
        Initialize the Canon class.

        Args:
            subgraph (ndarray): The adjacency matrix of the graph.
            classes (ndarray, optional): The colors (classes) of the subgraph nodes. Defaults to None.
        """
        self.subgraph = subgraph
        self.K = subgraph.shape[0]
        self.classes = classes if classes is not None else np.ones(self.K, dtype=int)
    
    def canon(self):
        """
        Perform a form of canonical labeling using NetworkX by sorting nodes and their neighborhoods.

        Returns:
            order (ndarray): The position vector of vertices in the new graph after "canonical labeling".
        """
        # Step 1: Reorder subgraph based on sorted classes
        classes, order = np.sort(self.classes), np.argsort(self.classes)
        subgraph1 = self.subgraph[np.ix_(order, order)]  # Reorder the adjacency matrix

        # Step 2: Convert adjacency matrix to a NetworkX graph
        G = nx.from_numpy_array(subgraph1)

        # Step 3: Sort nodes by degree and their neighborhood structure
        # Sort nodes by degree and then lexicographically by adjacency list (canonical form approximation)
        degree_sorted_nodes = sorted(G.nodes, key=lambda n: (G.degree(n), sorted(G.neighbors(n))))

        # Step 4: Map the sorted nodes back to the original order
        canonical_order = np.array(degree_sorted_nodes)
        order = np.array(order)

        # Create the final order array that matches the original labeling
        final_order = order[canonical_order]
        return final_order

