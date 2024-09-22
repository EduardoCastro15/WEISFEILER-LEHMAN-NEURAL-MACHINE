import numpy as np
from scipy.sparse import triu, csr_matrix, lil_matrix, find
import networkx as nx
import time

class DivideNet:
    def __init__(self, net, ratio_train):
        """
        Initialize DivideNet with the network (adjacency matrix) and the ratio for training links.
        
        Args:
            net (2D array): Adjacency matrix of the network.
            ratio_train (float): Ratio of links to use for training (between 0 and 1).
        """
        if isinstance(net, nx.Graph):
            # Convert networkx graph to sparse adjacency matrix
            self.net = csr_matrix(nx.to_scipy_sparse_array(net))
        else:
            # If net is already a matrix
            self.net = csr_matrix(net)
        self.ratio_train = ratio_train

    def divide(self):
        """
        Divide the network into training and testing sets.
        
        Returns:
            train (2D array): Training adjacency matrix.
            test (2D array): Testing adjacency matrix.
        """
        # Step 1: Convert to upper triangular matrix, removing self-loops
        net_upper = triu(self.net, k=1).tolil()  # Convert to upper triangle, excluding diagonal

        # Step 2: Calculate number of test links
        num_test_links = int(np.ceil((1 - self.ratio_train) * net_upper.nnz))

        # Step 3: Get the list of all edges (links)
        xindex, yindex, _ = find(net_upper)
        link_list = np.column_stack((xindex, yindex))

        # Step 4: Initialize the test adjacency matrix
        test = lil_matrix(self.net.shape)  # Use LIL format for efficient modification
        
        iteration = 0
        start_time = time.time()

        # Step 5: Randomly select links for the test set
        while test.nnz < num_test_links:
            if len(link_list) <= 2:
                break

            iteration += 1
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: test set size: {test.nnz}, remaining links: {len(link_list)}")
            
            # Randomly select a link from the link list
            index_link = np.random.randint(0, len(link_list))
            uid1, uid2 = link_list[index_link]
            
            # Remove the selected link from the network (set to 0)
            net_upper[uid1, uid2] = 0
            
            # Check connectivity of uid1 and uid2
            temp_vector = net_upper.getrow(uid1).toarray().flatten()  # Shape: (n,)
            sign = 0
            uid1_to_uid2 = temp_vector @ net_upper + temp_vector

            if uid1_to_uid2[uid2] > 0:
                sign = 1
            else:
                while np.any(np.clip(uid1_to_uid2, 0, 1) != temp_vector):
                    temp_vector = np.clip(uid1_to_uid2, 0, 1)  # Equivalent to MATLAB spones()
                    uid1_to_uid2 = temp_vector @ net_upper + temp_vector  # Matrix multiplication and addition
                    
                    if uid1_to_uid2[uid2] > 0:
                        sign = 1
                        break
            
            # Force sign to 1 to keep all selected links in the test set
            sign = 1
            
            if sign == 1:
                # Keep the link in the test set
                test[uid1, uid2] = 1
                link_list = np.delete(link_list, index_link, axis=0)
            else:
                # Restore the link to the network
                net_upper[uid1, uid2] = 1
        
        end_time = time.time()
        print(f"Finished link selection after {iteration} iterations in {end_time - start_time} seconds.")
        
        # Step 6: Create the train adjacency matrix
        net_upper = net_upper.tocsr()  # Convert back to CSR format for efficient operations
        test = test.tocsr()  # Convert the test matrix back to CSR format
        
        train = net_upper + net_upper.transpose()
        test = test + test.transpose()
        
        # Check for overlap between train and test sets
        assert train.multiply(test).nnz == 0, "Train and test sets should not overlap"

        return train, test
