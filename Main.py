import numpy as np
import time
import scipy.io
from scipy.sparse import csr_matrix

from WLNM import WLNM
from DataLoader import DataLoader
from DivideNet import DivideNet
from GraphVisualizer import GraphVisualizer

if __name__ == "__main__":
    datapath = 'data/'
    dataset_names = ['USAir.mat']  # List of datasets
    num_experiments = 1
    ratio_train = 0.9
    methods = [1]  # 1: WLNM
    num_in_each_method = [1]
    auc_for_dataset = []

    start_time = time.time()

    for ith_data, dataset_name in enumerate(dataset_names):
        print(f"Processing the {ith_data+1}th dataset: {dataset_name}")

        """
        Importing the dataset
        """
        # Specify the file path (can be either .mat or .csv)
        file_path = f"{datapath}/{dataset_name}"

        # Create an instance of DataLoader and load the data
        net = DataLoader(file_path).load_data()
        
        # Visualize network
        GraphVisualizer(net).draw_graph()

        # Initialize AUC storage
        num_of_methods = sum(num_in_each_method[i] for i in methods)
        auc_of_all_predictors = np.zeros((num_experiments, num_of_methods))
        predictors_name = []

        for ith_experiment in range(num_experiments):
            if ith_experiment % 10 == 0:
                print(f"Running experiment {ith_experiment}...")

            # Divide network into train and test
            train, test = DivideNet(net, ratio_train).divide()
            
            # Convert to sparse matrix format for computational efficiency
            train = csr_matrix(train)
            test = csr_matrix(test)
            
            # Forcing symmetrical adjacency matrices
            train = train + train.transpose()
            test = test + test.transpose()
            
            # AUC storage for current experiment
            ith_auc_vector = []
        
            # Run WLNM (if selected)
            if 1 in methods:
                print("Running WLNM...")
                temp_auc = WLNM(train, test, 10, ith_experiment).run()
                ith_auc_vector.append(temp_auc)
            
            # Save AUC results for current experiment
            auc_of_all_predictors[ith_experiment, :] = ith_auc_vector
    
        # Compute average and variance of AUC for all experiments
        avg_auc = np.mean(auc_of_all_predictors, axis=0)
        auc_for_dataset.append(avg_auc)
        var_auc = np.var(auc_of_all_predictors, axis=0)
        
        # Write results to file
        respath = f"{datapath}/result/{dataset_name}_res.txt"
        with open(respath, 'w') as res_file:
            res_file.write(f"{predictors_name}\n")
            np.savetxt(respath, np.vstack([avg_auc, var_auc]), delimiter='\t', fmt='%.4f')
    
    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time} seconds")
