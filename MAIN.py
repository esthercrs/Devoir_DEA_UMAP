print("Loading imports...")
import os
from Generateur_Gaussian_Cluster import Gaussian_Cluster_generator
from Generateur_Uniform import Uniform_generator
from UMAP import UMAP_dimension_reduction
from metrics import deformation_quantif_metrics
print("Imports done.")

directory = os.getcwd() + '/'

print("Creation folders...")
DATA_FOLDER = directory + "DATA_GENERATION"
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)
print("Folders done.")

os.environ["DATA_GENERATION_PATH"] = directory + "DATA_GENERATION"

print("Exécution du premier script de génération de données...")
Gaussian_Cluster_generator()

print("Exécution du second script de génération de données...")
Uniform_generator()

print("Creation folders...")
UMAP_FOLDER = directory + "DATA_UMAP"
if not os.path.exists(UMAP_FOLDER):
    os.makedirs(UMAP_FOLDER)
print("Folders done.")

os.environ["DATA_UMAP_PATH"] = directory + "DATA_UMAP"

print("Exécution du script UMAP...")
UMAP_dimension_reduction(directory + "DATA_GENERATION/resultats_Gaussian/")
UMAP_dimension_reduction(directory + "DATA_GENERATION/resultats_Cluster/")
UMAP_dimension_reduction(directory + "DATA_GENERATION/resultats_Uniform/")

print("Creation folders...")
METRICS_FOLDER = directory + "METRICS/"
if not os.path.exists(METRICS_FOLDER):
    os.makedirs(METRICS_FOLDER)
print("Folders done.")

os.environ["DATA_METRICS_PATH"] = directory + "METRICS/"

print("Exécution du script metrics...")
deformation_quantif_metrics(directory + "DATA_UMAP/resultats_Gaussian/")
deformation_quantif_metrics(directory + "DATA_UMAP/resultats_Cluster/")
deformation_quantif_metrics(directory + "DATA_UMAP/resultats_Uniform/")

print("Scripts terminés.")
