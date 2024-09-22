import numpy as np
import networkx as nx
import pybliss as bliss

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
        Perform canonical labeling of the subgraph using the bliss library.

        Returns:
            order (ndarray): The position vector of vertices in the new graph after canonical labeling.
        """
        # Step 1: Reorder subgraph based on sorted classes
        classes, order = np.sort(self.classes), np.argsort(self.classes)
        subgraph1 = self.subgraph[np.ix_(order, order)]  # Reorder the adjacency matrix

        # Step 2: Convert adjacency matrix to NetworkX graph
        G = nx.from_numpy_array(subgraph1)

        # Step 3: Perform canonical labeling using bliss
        bliss_graph = bliss.Graph(G.number_of_nodes())
        for u, v in G.edges():
            bliss_graph.connect(u, v)
        
        # Get the canonical labeling
        canonical_labeling = bliss_graph.canonical_labeling()

        # Step 4: Map canonical labels back to the original order
        canonical_order = np.array(canonical_labeling)
        order = np.array(order)
        
        # Create the final order array that matches the original labeling
        final_order = order[canonical_order]
        return final_order

