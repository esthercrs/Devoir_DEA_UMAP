import numpy as np
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors
from UMAP import get_files
from csv import writer

def dist_matrix(X):
    """This function calculates a matrix d of distances between points of the high dimensional data. 
    d[i][j] returns the distance between points i and j.

    Args:
        X (pandas dataframe): dataframe containing the high dimensional data

    Returns:
        numpy array: distance matrix
    """
    if type(X) is not(np.ndarray):
        X = X.to_numpy()
    n_samples = X.shape[0]
    dist = - np.ones((n_samples, n_samples))
    for i in range(n_samples):
        x_i = X[i]
        for j in range(i+1,n_samples):
            x_j = X[j]
            dist[i][j] = np.linalg.norm(abs(x_i - x_j))
    return dist

def weight_matrix(d, alpha):
    """This function calculates a matrix w of weights based on a previously calculated distance matrix and a constant alpha. 
    w[i][j] returns the weight associated to the distance between points i and j.


    Args:
        d (numpy array): distance matrix
        alpha (float): constant
                       If alpha >= 1, small distances are favored
                       If 0 < alpha < 1, big distances are favored
                       If alpha = 0, neither big or small distances are favoured because all the weights are equal to 1

    Returns:
        numpy array: weight matrix
    """
    n_samples = np.shape(d)[0]
    w = - np.ones((n_samples, n_samples))
    for i in range(n_samples):
        for j in range (i+1,n_samples):
            if d[i][j] != 0:
                w[i][j] = 1 / d[i][j] ** alpha
            elif d[i][j] == 0:
                w[i][j] = 1
    return w

def stress_majo(X_HD, X_LD, alpha=2.0): 
    """This function calculates the normalized stress majorization of data.

    Args:
        X_HD (pandas dataframe): dataframe containing the high dimensional data
        X_LD (numpy array): array of the dimensionally reduced data
        alpha (float): constant used in the computing of the weight matrix
                       Defaults to 2.0 (in order to favor small distances)

    Returns:
        float: normalized stress majorization
    """
    d = dist_matrix(X_HD)
    w = weight_matrix(d, alpha)
    n_samples = np.shape(d)[0]
    n_pairs = 0
    s = 0
    for i in range(n_samples):
        for j in range(i+1,n_samples):
            n_pairs += 1
            s += w[i][j] * (np.linalg.norm(X_LD[i] - X_LD[j]) - d[i][j]) ** 2
    return s / n_pairs

def k_neighbors(X, k):
    """This function finds, for each point of the input data, the indices of its k nearest neighbors.

    Args:
        X (numpy array or pandas dataframe): high dimensional or dimensionally reduced data
        k (int): number of neighbors of a point to consider

    Returns:
        numpy array: array of the indices of the k nearest neighbors for each point of the input data
    """
    if type(X) is not(np.ndarray):
        X = X.to_numpy()
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(X)
    indices = nbrs.kneighbors(X)[1]
    return indices

def jaccard_neighbors(X_HD, X_LD, k=7):
    """This function calculates the Jaccard index of the k nearest neighbors of each point and then computes the mean Jaccard index.

    Args:
        X_HD (pandas dataframe): dataframe containing the high dimensional data
        X_LD (numpy array): array of the dimensionally reduced data
        k (int): number of neighbors to consider
                 Defaults to 7

    Returns:
        float: mean Jaccard index of the k nearest neighbors (in %)
    """
    nbrs_LD = k_neighbors(X_LD, k)
    nbrs_HD = k_neighbors(X_HD, k)
    list_jaccard = []
    for i in range(np.shape(nbrs_HD)[0]):
        intersection = np.intersect1d(nbrs_LD[i], nbrs_HD[i])
        union = np.union1d(nbrs_LD[i], nbrs_HD[i])
        jaccard_index = len(intersection) / len(union)
        list_jaccard.append(jaccard_index)
    return sum(list_jaccard) * 100 / len(list_jaccard)

def get_params_np(filename):
    """This function is used to get the parameters used to generate the high dimension data and 
    those used for the UMAP dimension reduction which are contained in the file name. 
    This is done thanks to the split() method combined with the character "_" that separates each parameter.

    Args:
        filename (str): file name of the .npy file containing the dimensionally reduced data

    Returns:
        dict: dictionnary whose keys are parameters names and values are parameters values
    """
    name_split = filename[:-4].split('_')
    dict_params = {'distrib' : name_split[0]}
    if dict_params['distrib'] == 'uniform':
        dict_params['n_samples'] = int(name_split[1])
        dict_params['n_features'] = int(name_split[2])
        dict_params['k'] = int(name_split[3])
        dict_params['n_neighbors'] = int(name_split[4])
        dict_params['min_dist'] = float(name_split[5])
        dict_params['n_components'] = int(name_split[6])
        dict_params['metric'] = name_split[7]
    else:
        dict_params['n_samples'] = int(name_split[1])
        dict_params['cluster_std'] = float(name_split[2])
        dict_params['nb_centers'] = int(name_split[3])
        dict_params['n_features'] = int(name_split[4])
        dict_params['k'] = int(name_split[5])
        dict_params['n_neighbors'] = int(name_split[6])
        dict_params['min_dist'] = float(name_split[7])
        dict_params['n_components'] = int(name_split[8])
        dict_params['metric'] = name_split[9]
    return dict_params

def deformation_quantif_metrics(directory):
    """This function creates a .csv file whose columns are the parameters contained in the name of 
    a dimensionally reduced data .npy file and the metrics calculated for these data. 
    For each file in the directory, the dimension reduction deformation metrics
    (mean Jaccard index of nearest neighbors and stress majorization) 
    are calculated and the parameters contained in the file name are collected. 
    Each row of the .csv file, contains these informations for one .npy file.

    Args:
        directory (str): path to the directory where the dimensionally reduced data files are located
    """
    filenames = get_files(directory)
    output_folder_metrics = os.environ.get("DATA_METRICS_PATH", "")
    if not os.path.exists(output_folder_metrics):
        os.makedirs(output_folder_metrics)
    params = get_params_np(filenames[0])
    cols = list(params.keys()) + ['stress_majo', 'knn_jaccard_index']
    df_metrics = pd.DataFrame(columns=cols)
    file = output_folder_metrics + 'metrics_{}.csv'.format(params['distrib'])
    df_metrics.to_csv(file)
    for filename in filenames:
        np_params = get_params_np(filename)
        row = list(np_params.values())
        X_LD = np.load(directory + filename)
        folder = os.environ.get("DATA_GENERATION_PATH", "")
        if row[0] == 'uniform':
            X_HD = pd.read_csv(folder + '/resultats_Uniform/uniform_0_1_{}_{}_{}.csv'.format(row[1], row[2], row[3]))
        elif row[0] == 'gaussian':
            if row[2] == 1:
                X_HD = pd.read_csv(folder + '/resultats_Gaussian/{}_{}_{}_{}_{}_{}.csv'.format(row[0], row[1], int(row[2]), row[3], row[4], row[5])).drop(columns = 'cluster')
            else:
                X_HD = pd.read_csv(folder + '/resultats_Gaussian/{}_{}_{}_{}_{}_{}.csv'.format(row[0], row[1], row[2], row[3], row[4], row[5])).drop(columns = 'cluster')
        else:
            if row[2] == 1:
                X_HD = pd.read_csv(folder + '/resultats_Cluster/{}_{}_{}_{}_{}_{}.csv'.format(row[0], row[1], int(row[2]), row[3], row[4], row[5])).drop(columns = 'cluster')
            else:
                X_HD = pd.read_csv(folder + '/resultats_Cluster/{}_{}_{}_{}_{}_{}.csv'.format(row[0], row[1], row[2], row[3], row[4], row[5])).drop(columns = 'cluster')        
        stress = stress_majo(X_HD, X_LD)
        jaccard = jaccard_neighbors(X_HD, X_LD)
        row = row + [stress, jaccard]
        with open(file, 'a', newline='') as f:
            writer_object = writer(f)
            writer_object.writerow(row)
            f.close()

    

