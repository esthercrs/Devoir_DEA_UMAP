# Data loading
import numpy as np
import pandas as pd
import os
import itertools

def data_generation(low, high, size):
    """This function generates normalized synthetic data with a uniform distribution.

    Args:
        low (float): lower limit of the output interval of the generated data
                     All feature values generated will be greater than or equal to low
                     Defaults to 0.0
        high (float): upper limit of the output interval of the generated data
                      All values generated will be less than or equal to high 
                      Defaults to 1.0
        size (tuple of length 2): tuple (number of samples, number of features) setting the size of the high dimensional generated data

    Returns:
        numpy array: array of the normalized generated samples
    """
    Uniform_array = np.random.uniform(low=low, high=high, size=size)
    return Uniform_array

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
    names_columns = feature_columns 
    return names_columns

def file_names(output_folder, low, high, n_samples, n_features, df, k):
    """This function creates a .csv file containing the generated high dimensional data.
    The file name specifies the type of distribution used and the other parameters.

    Args:
        output_folder (str): path to the directory in which to save the generated data
        low (float): lower limit of the output interval of the generated data
        high (float): upper limit of the output interval of the generated data
        n_samples (int): number of samples 
        n_features (int): number of features for each sample
        df (pandas dataframe): dataframe containing the generated high dimensional data
        k (int): number of repetitions of data generated with the previous parameters
                 Defaults to 5
    """
    file_path = os.path.join(output_folder,"uniform_{}_{}_{}_{}_{}.csv".format(low, high, n_samples, n_features, k))
    df.to_csv(file_path, index=False)

def parameters_testing(low, high, n_samples, n_features, output_folder, k=5) :
    """This function generates the high dimensional data with all the possible combinations of parameters values.
    Then, the data is saved in a .csv file.

    Args:
        low (float): lower limit of the output interval of the generated data
        high (float): upper limit of the output interval of the generated data
        n_samples (int): number of samples
        n_features (int): number of features for each sample
        output_folder (str): path to the directory in which to save the generated data
        k (int): number of repetitions of data generated with the previous parameters
                 Defaults to 5
    """
    SxF = cartesian_product(n_samples, n_features)
    for i in range(0, k):
        for parametres in SxF : 
            s,f = parametres
            size = (s, f)
            Uniform_array = data_generation(low, high, size)
            nc = columns_names(f)
            df = pd.DataFrame(Uniform_array, columns=nc)
            file_names(output_folder, low, high, s, f, df, i)

def cartesian_product(n_samples, n_features):
    """This function returns the cartesian product of the parameters values.

    Args:
        n_samples (int): number of samples
        n_features (int): number of features for each sample

    Returns:
        set: set containing the cartesian product of the parameters
             Each element of the set is a tuple of length 2 representing a combination of (n_samples, n_features) values
    """
    return set(itertools.product(n_samples, n_features))

def Uniform_generator(low=0.0, high=1.0, n_samples=[i for i in range(100,5001,100)], n_features=[2**j for j in range(2,7)]):
    """This function first creates the folder that will contain the generated data. 
    Then, the parameters_testing function is launched.

    Args:
        low (float): lower limit of the output interval of the generated data
                     Defaults to 0.0
        high (float): upper limit of the output interval of the generated data
                      Defaults to 1.0
        n_samples (int): number of samples
                         Defaults to [i for i in range(100,5001,100)]
        n_features (int): number of features for each sample
                          Defaults to [2**j for j in range(2,7)]
    """
    output_folder = os.path.join(os.environ.get("DATA_GENERATION_PATH", ""), "resultats_Uniform")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    parameters_testing(low, high, n_samples, n_features, output_folder)