import numpy as np
from scipy.sparse.csgraph import connected_components
import networkx as nx
from networkx import Graph, set_node_attributes, adjacency_matrix, relabel_nodes

from PaletteWL import PaletteWL
from Canon import Canon

class Graph2Vector:
    def __init__(self, pos, neg, A, K):
        """
        Initialize the Graph2Vector class.
        
        Args:
            pos (array): Indices of positive links.
            neg (array): Indices of negative links.
            A (2D array or sparse matrix): The observed graph's adjacency matrix.
            K (int): The number of nodes in each link's subgraph.
        """
        self.pos = pos  # Positive links
        self.neg = neg  # Negative links
        self.A = A  # adjacency matrix of the graph
        self.K = K  # number of nodes in the subgraph

    def graph2vector(self):
        """
        Convert links' enclosing subgraphs (both positive and negative) into vectors.
        
        Returns:
            data (array): The constructed feature vectors, each row is a link's vector representation.
            label (array): A column vector of links' labels (1 for positive, 0 for negative).
        """
        all_links = np.vstack([self.pos, self.neg])  # Combine positive and negative links
        pos_size = self.pos.shape[0]
        neg_size = self.neg.shape[0]
        
        # Generate labels: 1 for positive links, 0 for negative links
        label = np.hstack([np.ones(pos_size), np.zeros(neg_size)])
        
        # Dimension of the data vectors (K*(K-1)/2)
        d = int(self.K * (self.K - 1) / 2)
        all_size = pos_size + neg_size
        
        # Initialize the data matrix to hold the vector representations of the links
        data = np.zeros((all_size, d))
        
        print("Subgraph Pattern Encoding Begins...")
        
        # Loop through all links (both positive and negative)
        for i in range(all_size):
            ind = all_links[i, :]
            sample = self.subgraph2vector(ind)
            data[i, :] = sample
            
            # Display progress
            if i % (all_size // 10) == 0 and i > 0:
                print(f"Subgraph Pattern Encoding Progress: {int(i / all_size * 100)}%")
        
        return data, label

    def subgraph2vector(self, ind):
        """
        Extract the enclosing subgraph for a link and convert it into a vector representation.
        
        Args:
            ind (tuple): Indices of the link (i, j).
        
        Returns:
            sample (array): The vector representation of the enclosing subgraph.
        """
        D = int(self.K * (self.K - 1) / 2)  # Length of the output vector
        i, j = ind  # The link (i, j) for which we need the subgraph
        
        # Initialize nodes and distances
        nodes = np.array([i, j])  # Initial set of nodes (the two nodes in the link)
        nodes_dist = np.array([0, 0])  # Distance to the initial nodes (0 distance)
        links = np.array([[i, j]])  # Initial link
        links_dist = np.array([0])  # Distance for the initial link
        dist = 0  # Initialize distance
        fringe = np.array([[i, j]])  # Initialize the fringe (neighbors to explore)
        
        while True:
            dist += 1
            # Get neighbors for all links in the fringe
            new_fringe = self.neighbors(fringe, self.A)
            
            # Apply setdiff to remove already processed links
            fringe = self.setdiff(new_fringe, links)
            
            # Check if no more new neighbors
            if fringe.size == 0:
                # Build subgraph for the current set of nodes
                subgraph = self.A[np.ix_(nodes, nodes)]
                subgraph[0, 1] = subgraph[1, 0] = 0  # Remove the original link information
                break
            
            # Find new nodes that haven't been added to the subgraph
            new_nodes = np.unique(fringe.flatten())
            new_nodes = np.setdiff1d(new_nodes, nodes)  # Remove nodes already in the subgraph
            
            # Apply setdiff to remove nodes already in the subgraph (Pending to confirm this line performes the same operation as above)
            #new_nodes = self.setdiff(new_nodes, nodes)
            
            # Update the nodes and distances
            nodes = np.hstack([nodes, new_nodes])
            nodes_dist = np.hstack([nodes_dist, np.ones(len(new_nodes)) * dist])
            
            # Update links and distances
            links = np.vstack([links, fringe])
            links_dist = np.hstack([links_dist, np.ones(fringe.shape[0]) * dist])
            
            # Check if we have enough nodes for the subgraph
            if len(nodes) >= self.K:
                nodes = nodes[:self.K]  # Limit the nodes to K
                subgraph = self.A[np.ix_(nodes, nodes)]
                subgraph[0, 1] = subgraph[1, 0] = 0  # Remove the original link information
                break
        
        # Step: Calculate link-weighted subgraph
        links_ind = np.ravel_multi_index((links[:, 0], links[:, 1]), self.A.shape)
        A_copy = self.A / (dist + 1) # if a link between two existing nodes < dist+1, it must be in 'links'. The only links not in 'links' are the dist+1 links between some farthest nodes in 'nodes', so here we weight them by dist+1
        A_copy.ravel()[links_ind] = 1.0 / links_dist  # Weight by inverse of distance
        
        # Keep the minimum distance for each edge
        A_copy_u = np.maximum(np.triu(A_copy, 1), np.tril(A_copy, -1).T) # for links (i, j) and (j, i), keep the smallest dist
        lweight_subgraph = A_copy_u + A_copy_u.T
        
        # Step: Perform graph labeling and reorder the subgraph
        order = self.g_label(subgraph)
        
        if len(order) > self.K: # if size > K, keep only the top-K vertices and reorder
            order = order[:self.K]
            subgraph = subgraph[np.ix_(order, order)]
            lweight_subgraph = lweight_subgraph[np.ix_(order, order)]
            order = self.g_label(subgraph)
        
        # Step: Generate enclosing subgraph vector representation
        ng2v = 2  # Method for transforming a g_labeled subgraph to vector
        if ng2v == 1:
            # Simplest way: one-dimensional vector by flattening the adjacency matrix
            psubgraph = subgraph[np.ix_(order, order)]
            sample = psubgraph[np.triu_indices_from(psubgraph, k=1)]
            sample[0] = np.finfo(float).eps  # Avoid empty vector in libsvm format
        elif ng2v == 2:
            # Link distance-weighted adjacency matrix
            plweight_subgraph = lweight_subgraph[np.ix_(order, order)]
            sample = plweight_subgraph[np.triu_indices_from(plweight_subgraph, k=1)]
            sample[0] = np.finfo(float).eps
        
        # Add dummy nodes if not enough nodes extracted in the subgraph
        if len(sample) < D:
            sample = np.concatenate([sample, np.zeros(D - len(sample))])

        return sample

    @staticmethod
    def neighbors(fringe, A):
        """
        Find the neighbor links of all links in fringe from adjacency matrix A.
        
        Args:
            fringe (array): List of edges (pairs of nodes) whose neighbors we want to find.
            A (2D array): Adjacency matrix of the graph.
            
        Returns:
            N (array): List of neighboring edges (pairs of nodes) for all links in fringe.
        """
        N = []
        for edge in fringe:
            i, j = edge
            
            # Find neighbors of node i (row i)
            row_neighbors = np.where(A[i, :] > 0)[0]  # Nodes connected to i
            col_neighbors = np.where(A[:, j] > 0)[0]  # Nodes connected to j
            
            # Add new edges (i, row neighbors) and (col neighbors, j)
            new_edges = np.vstack([
                np.column_stack((i * np.ones(len(row_neighbors), dtype=int), row_neighbors)),
                np.column_stack((col_neighbors, j * np.ones(len(col_neighbors), dtype=int)))
            ])
            
            # Append and deduplicate while preserving order
            if N == []:
                N = new_edges
            else:
                N = np.vstack([N, new_edges])
            
            # Eliminate repeated edges, keep in order
            N = np.unique(N, axis=0)

        return N

    @staticmethod
    def setdiff(fringe, links):
        """
        Remove rows from 'fringe' that are already present in 'links'.
        
        Args:
            fringe (ndarray): Array of edges (rows of node pairs).
            links (ndarray): Array of already processed edges (rows of node pairs).
            
        Returns:
            ndarray: Updated 'fringe' with rows from 'links' removed.
        """
        if fringe.size == 0:
            return fringe
        if links.size == 0:
            return fringe

        # Use numpy's setdiff1d to find rows that are in 'fringe' but not in 'links'
        dtype = [('f{}'.format(i), fringe.dtype) for i in range(fringe.shape[1])]
        
        # Convert arrays to structured arrays so that rows are compared as whole
        fringe_structured = fringe.view(dtype)
        links_structured = links.view(dtype)
        
        # Perform set difference on the structured arrays
        diff_structured = np.setdiff1d(fringe_structured, links_structured)
        
        # Convert back to the original array form
        return diff_structured.view(fringe.dtype).reshape(-1, fringe.shape[1])

    @staticmethod
    def g_label(subgraph, p_mo=7):
        """
        Impose a vertex order for an enclosing subgraph using graph labeling.
        
        Args:
            subgraph (ndarray): The adjacency matrix of the subgraph.
            p_mo (int): Palette method option (default: 7, palette_wl with initial colors).
        
        Returns:
            order (array): The vertex order of the graph based on labeling.
        """
        K = subgraph.shape[0]
        
        # Calculate initial colors based on geometric mean distance to the link
        dist_to_1 = Graph2Vector.get_shortest_path_lengths(subgraph, 0, K)
        dist_to_2 = Graph2Vector.get_shortest_path_lengths(subgraph, 1, K)
        
        # dist_to_1 = np.array([dist_to_1.get(i, 2 * K) for i in range(K)])  # Replace inf with 2 * K
        # dist_to_2 = np.array([dist_to_2.get(i, 2 * K) for i in range(K)])  # Replace inf with 2 * K
        dist_to_1 = np.array([dist_to_1[i] if i in dist_to_1 else 2 * K for i in range(K)])
        dist_to_2 = np.array([dist_to_2[i] if i in dist_to_2 else 2 * K for i in range(K)])

        
        avg_dist = np.sqrt(dist_to_1 * dist_to_2)
        _, _, avg_dist_colors = np.unique(avg_dist, return_inverse=True, return_counts=False)
        
        # Switch different graph labeling methods
        """
        if p_mo == 1:
            classes = wl_string_lexico(subgraph)
            order = canon(subgraph, classes)
        elif p_mo == 2:
            classes = wl_hashing(subgraph)
            order = canon(subgraph, classes)
        elif p_mo == 3:
            classes = wl_string_lexico(subgraph, avg_dist_colors)
            order = canon(subgraph, classes)
        elif p_mo == 4:
            classes = wl_hashing(subgraph, avg_dist_colors)
            order = canon(subgraph, classes)
        elif p_mo == 5:
            order = canon(subgraph, np.ones(K))
        elif p_mo == 6:
            order = np.arange(1, K + 1)

        elif p_mo == 8:
            order = np.random.permutation(K)
        """
        if p_mo == 7:
            classes = PaletteWL(subgraph, avg_dist_colors).palette_wl()
            order = Canon(subgraph, classes).canon()
        
        return order

    @staticmethod
    def get_shortest_path_lengths(subgraph, source, K):
        """
        Get the shortest path lengths from a source node to all other nodes in an undirected graph.
        
        Args:
            subgraph (ndarray): The adjacency matrix of the subgraph.
            source (int): The source node index.
            K (int): The size of the subgraph.
        
        Returns:
            dist_to_source (ndarray): Array of shortest path lengths, with '2 * K' for unreachable nodes.
        """
        G = nx.Graph(subgraph)  # Create an undirected graph from the adjacency matrix
        path_lengths = nx.single_source_shortest_path_length(G, source)  # Get shortest path lengths
        
        # Initialize distances with 2 * K (unreachable nodes)
        dist_to_source = np.full(K, 2 * K)
        
        # Fill in the actual distances
        for node, dist in path_lengths.items():
            dist_to_source[node] = dist
        
        return dist_to_source
