import numpy as np
from scipy.sparse import triu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim

from SampleNeg import SampleNeg
from Graph2Vector import Graph2Vector

class WLNM:
    def __init__(self, train, test, K=10, ith_experiment=1):
        """
        Initialize the Weisfeiler-Lehman Neural Machine (WLNM) algorithm.
        
        Args:
            train (sparse matrix): The training adjacency matrix (1 for link, 0 otherwise).
            test (sparse matrix): The testing adjacency matrix (1 for link, 0 otherwise).
            K (int): Number of vertices in an enclosing subgraph.
            ith_experiment (int): Experiment index for parallel computing (default is 1).
        """
        self.train = triu(train, k=1)  # upper triangular part
        self.test = triu(test, k=1)    # upper triangular part
        self.K = K
        self.ith_experiment = ith_experiment

    def run(self):
        """
        Main function to run the WLNM algorithm and compute AUC.
        
        Returns:
            auc (float): The AUC score of the model.
        """
        # Step 1: Sample positive and negative links
        train_pos, train_neg, test_pos, test_neg = SampleNeg(self.train, self.test, 2, 1, True).sample()

        # Step 2: Convert graphs to feature vectors
        train_data, train_label = Graph2Vector(train_pos, train_neg, self.train, self.K).graph2vector()
        test_data, test_label = Graph2Vector(test_pos, test_neg, self.train, self.K).graph2vector()

        # Step 3: Train a model (Feedforward neural network)
        model_type = 2  # Logistic regression

        if model_type == 1:
            # Logistic Regression using scikit-learn
            clf = LogisticRegression()
            clf.fit(train_data, train_label)
            test_preds = clf.predict_proba(test_data)[:, 1]
        
        elif model_type == 2:
            # Feedforward neural network using PyTorch
            input_size = train_data.shape[1]
            model = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Convert data to PyTorch tensors
            train_tensor = torch.FloatTensor(train_data)
            train_labels_tensor = torch.FloatTensor(train_label).unsqueeze(1)
            test_tensor = torch.FloatTensor(test_data)
            test_labels_tensor = torch.FloatTensor(test_label).unsqueeze(1)

            # Train the neural network
            epochs = 10
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(train_tensor)
                loss = criterion(outputs, train_labels_tensor)
                loss.backward()
                optimizer.step()
            
            # Predict on the test set
            test_preds = model(test_tensor).detach().numpy().flatten()

        # Step 4: Compute AUC score
        auc = roc_auc_score(test_label, test_preds)
        print(f"AUC Score: {auc}")
        return auc
