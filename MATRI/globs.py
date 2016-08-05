###################################################################
# Parameters
###################################################################


###################### DATASETS######################

# DATASET_NAME = "dataset/advogato-graph-2000-02-25.dot"
# RATING_MAP = {'Observer': 0.1, 'Apprentice': 0.4, 'Journeyer': 0.7,
#               'Master': 0.9}

DATASET_NAME = "dataset/advogato-graph-2011-06-23.dot"
RATING_MAP = {'"Observer"': 0.1, '"Apprentice"': 0.4, '"Journeyer"': 0.7,
              '"Master"': 0.9}

#####################################################

GLOBAL_t = 6                # Maximum propagation step

GLOBAL_p = 3                # Total number of bias factors
GLOBAL_r = 10              # Total number latent factors
GLOBAL_l = 10              # ??

GLOBAL_max_itr = 1000       # Max itreration untill convergence.

GLOBAL_lamda = 1.1         # Regularization parameter for parameter updation.


# Names of Files for saving
FILE_DIR = "Saved/"

FILE_Z_train = FILE_DIR + "Z_train"
FILE_Z_test = FILE_DIR + "Z_test"
FILE_Z = FILE_DIR + "Z_full"
FILE_RMSE = FILE_DIR + "RMSE"

# GLOBAL_EPS = np.finfo(float).eps
GLOBAL_EPS = 0.000001
GLOBAL_FACTORIZATION_MAX_ITER = 200

RANDOM_SEED_FACTORIZATION = 49

###################################################################
