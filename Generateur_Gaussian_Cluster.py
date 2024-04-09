# Data loading
from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
import os
import itertools
from sklearn.preprocessing import MinMaxScaler


def data_generation(n_samples, cluster_std, nb_centers, n_features, random_state):
    """This function generates normalized synthetic data either with a cluster distribution 
    or a gaussian distribution (by setting the number of clusters to 1).

    Args:
        n_samples (int): number of samples (equally distributed between clusters)
        cluster_std (float): clusters standard deviation
        nb_centers (int): number of clusters 
        n_features (int): number of features for each sample
        random_state (int): determines random number generation for dataset creation

    Returns:
        numpy array: array Xnorm of the normalized generated samples
        numpy array: array y of the cluster labels of each of the samples
    """
    X, y = make_blobs(n_samples=n_samples, cluster_std=cluster_std, centers=nb_centers, n_features=n_features, random_state=random_state, return_centers=False)
    scaler = MinMaxScaler()
    Xnorm = scaler.fit_transform(X) 
    return Xnorm, y

def columns_names(n_features):
    """This function defines the columns names for the .csv file of the generated data. 
    Depending on the number of features N, we have N feature columns and 
    a last column specifying the cluster to which the sample belongs.

    Args:
        n_features (int): number of features for each sample

    Returns:
        list: list of columns names to be included in the generated data .csv file
    """
    feature_columns=[]
    for i in range(1, n_features+1):   
        feature_columns.append('feature_{}'.format(i))
    names_columns = feature_columns + ['cluster']
    return names_columns

def file_names(output_folder, n_samples, cluster_std, nb_centers, n_features, df, k):
    """This function creates a .csv file containing the generated high dimensional data.
    The file name specifies the type of distribution used and the other parameters.

    Args:
        output_folder (str): path to the directory in which to save the generated data
        n_samples (int): number of samples (equally distributed between clusters)
        cluster_std (float): clusters standard deviation
        nb_centers (int): number of clusters
        n_features (int): number of features for each sample
        df (pandas dataframe): dataframe containing the generated high dimensional data
        k (int): number of repetitions of data generated with the previous parameters
                 Defaults to 5
    """
    if nb_centers == 1: 
        file_path = os.path.join(output_folder[0],"gaussian_{}_{}_{}_{}_{}.csv".format(n_samples, cluster_std, nb_centers, n_features, k))
    else : 
        file_path = os.path.join(output_folder[1],"cluster_{}_{}_{}_{}_{}.csv".format(n_samples, cluster_std, nb_centers, n_features, k))
    df.to_csv(file_path, index=False)

def parameters_testing(n_samples, cluster_std, nb_centers, n_features, random_state, output_folder, k=5) :
    """This function generates the high dimensional data with all the possible combinations of parameters values.
    Then, the data is saved in a .csv file.

    Args:
        n_samples (int): number of samples (equally distributed between clusters)
        cluster_std (float): clusters standard deviation
        nb_centers (int): number of clusters
        n_features (int): number of features for each sample
        random_state (int): determines random number generation for dataset creation
        output_folder (str): path to the directory in which to save the generated data
        k (int): number of repetitions of data generated with the previous parameters
                 Defaults to 5
    """
    SxSTDxCxF = cartesian_product(n_samples, cluster_std, nb_centers, n_features)
    for i in range(0, k):
        for parametres in SxSTDxCxF: 
            s, std, c, f = parametres
            X, y = data_generation(s, std, c, f, random_state)
            array_numpy = np.concatenate([X, y.reshape(-1, 1).astype(int)], axis=1) 
            nc = columns_names(f)
            df = pd.DataFrame(array_numpy, columns=nc)
            file_names(output_folder, s, std, c, f, df, i)

def cartesian_product(n_samples, cluster_std, nb_centers, n_features):
    """This function returns the cartesian product of the parameters values.

    Args:
        n_samples (int): number of samples (equally distributed between clusters)
        cluster_std (float): clusters standard deviation
        nb_centers (int): number of clusters
        n_features (int): number of features for each sample

    Returns:
        set: set containing the cartesian product of the parameters
             Each element of the set is a tuple of length 4 representing a combination of (n_samples, cluster_std, nb_centers, n_features) values
    """
    return set(itertools.product(n_samples, cluster_std, nb_centers, n_features))

def Gaussian_Cluster_generator(n_samples=[i for i in range(100,5001,100)], cluster_std=[0.0, 0.5, 1], nb_centers=[1,4,10], n_features=[2**j for j in range(2,7)], random_state=0):
    """This function first creates the folder that will contain the generated data. 
    Then, the parameters_testing function is launched.

    Args:
        n_samples (int): number of samples (equally distributed between clusters)
                         Defaults to [i for i in range(100,5001,100)]
        cluster_std (float): clusters standard deviation 
                             Defaults to [0.0, 0.5, 1]
        nb_centers (int): number of clusters
                          Defaults to [1,4,10]
        n_features (int): number of features for each sample
                          Defaults to [2**j for j in range(2,7)]
        random_state (int): determines random number generation for dataset creation
                            Defaults to 0
    """
    output_folder_Gaussian = os.path.join(os.environ.get("DATA_GENERATION_PATH", ""), "resultats_Gaussian")
    if not os.path.exists(output_folder_Gaussian):
        os.makedirs(output_folder_Gaussian)

    output_folder_Cluster = os.path.join(os.environ.get("DATA_GENERATION_PATH", ""), "resultats_Cluster")
    if not os.path.exists(output_folder_Cluster):
        os.makedirs(output_folder_Cluster)
    
    output_folder = [output_folder_Gaussian, output_folder_Cluster]

    parameters_testing(n_samples, cluster_std, nb_centers, n_features, random_state, output_folder)