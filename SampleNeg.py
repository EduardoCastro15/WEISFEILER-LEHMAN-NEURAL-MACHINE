import numpy as np
from scipy.sparse import triu, find

class SampleNeg:
    def __init__(self, train, test, k=1, portion=1, evaluate_on_all_unseen=True):
        """
        Initialize the SampleNeg class with the training and testing adjacency matrices.
        
        Args:
            train (sparse matrix): Training positive adjacency matrix.
            test (sparse matrix): Testing positive adjacency matrix.
            k (int): Factor for how many times more negative links to sample compared to positive links.
            portion (float): Portion of the sampled links to return.
            evaluate_on_all_unseen (bool): Whether to consider all unseen links as negative testing links.
        """
        self.train = train
        self.test = test
        self.k = k
        self.portion = portion
        self.evaluate_on_all_unseen = evaluate_on_all_unseen
    
    def sample(self):
        """
        Sample negative links for the training and testing sets.
        
        Returns:
            train_pos (array): Positive training links (edges).
            train_neg (array): Negative training links (non-edges).
            test_pos (array): Positive testing links (edges).
            test_neg (array): Negative testing links (non-edges).
        """
        n = self.train.shape[0]  # Number of nodes

        # Find the positive training links
        train_pos = np.column_stack(find(self.train))

        # Find the positive testing links
        test_pos = np.column_stack(find(self.test))

        train_size = len(train_pos)
        test_size = len(test_pos)

        # Combine train and test for the full network
        if self.test is None or self.test.nnz == 0:
            net = self.train
        else:
            net = self.train + self.test

        # Ensure that train and test do not overlap (assert no double-counting of edges)
        assert (self.train.multiply(self.test)).nnz == 0, "Train and test sets should not overlap"

        # Negative network (non-edges)
        neg_net = np.triu(-(net.toarray() - 1), k=1)  # Upper triangular to avoid duplicates
        neg_links = np.column_stack(np.where(neg_net > 0))  # Find the negative links

        # Sample negative links
        if self.evaluate_on_all_unseen:
            # All unseen links are used as negative test links
            test_neg = neg_links
            # Randomly select train negative links from all unknown links
            perm = np.random.permutation(neg_links.shape[0])
            train_neg = neg_links[perm[:self.k * train_size], :]
            # Remove the selected train negative links from the test negative links
            test_neg = np.delete(test_neg, perm[:self.k * train_size], axis=0)
        else:
            # Randomly sample negative links for training and testing
            nlinks = neg_links.shape[0]
            perm = np.random.permutation(nlinks)
            
            if self.k * (train_size + test_size) <= nlinks:
                # Sufficient negative links available
                train_ind = perm[:self.k * train_size]
                test_ind = perm[self.k * train_size:self.k * (train_size + test_size)]
            else:
                # Not enough negative links, so divide them proportionally
                ratio = train_size / (train_size + test_size)
                train_ind = perm[:int(ratio * nlinks)]
                test_ind = perm[int(ratio * nlinks):]
            
            train_neg = neg_links[train_ind, :]
            test_neg = neg_links[test_ind, :]

        # Optionally, return only a portion of the sampled links (for fitting into memory)
        if self.portion < 1:  # Portion is a fraction
            train_pos = train_pos[:int(np.ceil(train_pos.shape[0] * self.portion)), :]
            train_neg = train_neg[:int(np.ceil(train_neg.shape[0] * self.portion)), :]
            test_pos = test_pos[:int(np.ceil(test_pos.shape[0] * self.portion)), :]
            test_neg = test_neg[:int(np.ceil(test_neg.shape[0] * self.portion)), :]
        elif self.portion > 1:  # Portion is an integer (number of selections)
            train_pos = train_pos[:self.portion, :]
            train_neg = train_neg[:self.portion, :]
            test_pos = test_pos[:self.portion, :]
            test_neg = test_neg[:self.portion, :]

        return train_pos, train_neg, test_pos, test_neg
