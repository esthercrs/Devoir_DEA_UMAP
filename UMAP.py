import itertools
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
import umap.umap_ as umap


def get_files(directory):
    """This function is used to get the names of files contained in a directory.

    Args:
        directory (str): path to the directory 

    Returns:
        list: list of the files names
    """
    filenames = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            filenames.append(f.split('/')[-1])
    return filenames

def get_params(filename):
    """This function is used to get the parameters used to generate the high dimension data which are contained in the file name. 
    This is done thanks to the split() method combined with the character "_" that separates each parameter.

    Args:
        filename (str): generated high dimensional data file name

    Returns:
        dict: dictionnary whose keys are parameters names and values are parameters values used to generate the high dimension data 
    """
    name_split = filename[:-4].split('_')
    dict_params = {'distrib' : name_split[0]}
    if dict_params['distrib'] == 'uniform':
        dict_params['n_samples'] = int(name_split[3])
        dict_params['n_features'] = int(name_split[4])
        dict_params['k'] = int(name_split[5])
    else:
        dict_params['n_samples'] = int(name_split[1])
        dict_params['cluster_std'] = float(name_split[2])
        dict_params['nb_centers'] = int(name_split[3])
        dict_params['n_features'] = int(name_split[4])
        dict_params['k'] = int(name_split[5])
    return dict_params

def UMAP_red(X, n, d, c, m):
    """Thanks to UMAP, this function reduces the dimension of the data contained in the input dataframe.

    Args:
        X (pandas dataframe): dataframe containing the high dimensional data
        n (int): number of neighboring sample points used for manifold approximation
        d (float)): effective minimum distance between embedded points
        c (int): dimension of the space to embed into
        m (str): metric to use to compute distances in high dimensional space

    Returns:
        numpy array: array of the dimensionally reduced data
    """
    reducer = umap.UMAP(n_neighbors=n, min_dist=d, n_components=c, metric=m)
    data = X[list(X)].values
    return reducer.fit_transform(data)    

def np_file(UMAP_params, data_params, output_folder):
    """This function creates a .npy file name containing the parameters of the high dimensional data 
    and the parameters usedfor the UMAP dimension reduction. 
    Then, it concatenates this file name to the path in which we want to save the dimensionally reduced data.


    Args:
        UMAP_params (list): list of UMAP parameters used to reduce the high dimensional data
        data_params (dict): dictionnary whose keys are parameters names and values are parameters values used to generate the high dimension data 
        output_folder (str): path to the folder in which we want to save the .npy files

    Returns:  
        str: path to .npy file 
    """
    if data_params['distrib'] == 'uniform':
        np_path = os.path.join(output_folder,'uniform_{}_{}_{}_{}_{}_{}_{}.npy'.format(data_params['n_samples'],
                                                data_params['n_features'],
                                                data_params['k'],
                                                UMAP_params[0],
                                                UMAP_params[1],
                                                UMAP_params[2],
                                                UMAP_params[3]))
    else:
        np_path = os.path.join(output_folder,'{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.npy'.format(data_params['distrib'], 
                                            data_params['n_samples'],
                                            data_params['cluster_std'],
                                            data_params['nb_centers'],
                                            data_params['n_features'],
                                            data_params['k'],
                                            UMAP_params[0],
                                            UMAP_params[1],
                                            UMAP_params[2],
                                            UMAP_params[3]))
    return np_path

def UMAP_dimension_reduction(directory):
    """For each high dimensional data file in the directory and for each combination of the UMAP parameters, 
    this function applies UMAP method to reduce the data dimension. The result is normalized and then saved 
    as a .npy file in a specific output folder.

    Args:
        directory (str): path to the directory containing the generated high dimensional data
    """
    filenames = get_files(directory)
    for filename in filenames:
        data_params = get_params(filename)
        X_HD = pd.read_csv(directory+filename)
        n_neighbors=[5,10]
        min_dist=[0.1,0.5]
        n_components=[2]
        metric=['euclidean']
        UMAP_params = list(itertools.product(n_neighbors, min_dist, n_components, metric))
        if data_params['distrib'] == 'cluster': 
            output_folder_Cluster = os.path.join(os.environ.get("DATA_UMAP_PATH", ""), "resultats_Cluster")
            if not os.path.exists(output_folder_Cluster):
                os.makedirs(output_folder_Cluster)
            for p in UMAP_params:
                n, d, c, m = p
                np_path = np_file(p, data_params, output_folder_Cluster)
                if not os.path.exists(np_path):
                    X_LD = UMAP_red(X_HD.drop(columns = 'cluster'), n, d, c, m)
                    scaler = MinMaxScaler()
                    X_LD_norm = scaler.fit_transform(X_LD) 
                    np.save(np_path, X_LD_norm)
        elif data_params['distrib'] == 'gaussian':
            output_folder_Gaussian = os.path.join(os.environ.get("DATA_UMAP_PATH", ""), "resultats_Gaussian")
            if not os.path.exists(output_folder_Gaussian):
                os.makedirs(output_folder_Gaussian)
            for p in UMAP_params:
                n, d, c, m = p
                np_path = np_file(p, data_params, output_folder_Gaussian)
                if not os.path.exists(np_path):
                    X_LD = UMAP_red(X_HD.drop(columns = 'cluster'), n, d, c, m)
                    scaler = MinMaxScaler()
                    X_LD_norm = scaler.fit_transform(X_LD) 
                    np.save(np_path, X_LD_norm)
        elif data_params['distrib'] == 'uniform':
            output_folder_Uniform = os.path.join(os.environ.get("DATA_UMAP_PATH", ""), "resultats_Uniform")
            if not os.path.exists(output_folder_Uniform):
                os.makedirs(output_folder_Uniform)
            for p in UMAP_params:
                n, d, c, m = p
                np_path = np_file(p, data_params, output_folder_Uniform)
                if not os.path.exists(np_path):
                    X_LD = UMAP_red(X_HD, n, d, c, m)
                    scaler = MinMaxScaler()
                    X_LD_norm = scaler.fit_transform(X_LD) 
                    np.save(np_path, X_LD_norm)