"""
Author: Michael Ainsworth
"""


#%%
"""
Imports
"""
import numpy as np
import matplotlib.pyplot as plt
from random import sample
from tqdm.notebook import tqdm
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import openml
import itertools
import os.path
#%%
"""
Parameters for execution
"""

To_load=False
algorithms_to_rerun={'RF':0,'DN':0,'GBDT':1}
num_datasets=72 # originally 72

to_use_prev_parameters=True
to_replace_prev_parameters={'RF':0,'DN':0,'GBDT':1}
#%%


def load_cc18():
    """
    Import datasets from OpenML-CC18 dataset suite
    """
    X_data_list = []
    y_data_list = []
    dataset_name = []

    for task_num, task_id in enumerate(
        tqdm(openml.study.get_suite("OpenML-CC18").tasks)
    ):
        try:
            successfully_loaded = True
            dataset = openml.datasets.get_dataset(
                openml.tasks.get_task(task_id).dataset_id
            )
            dataset_name.append(dataset.name)
            X, y, is_categorical, _ = dataset.get_data(
                dataset_format="array", target=dataset.default_target_attribute
            )
            _, y = np.unique(y, return_inverse=True)
            X = np.nan_to_num(X)
        except TypeError:
            successfully_loaded = False
        if successfully_loaded and np.shape(X)[1] > 0:
            X_data_list.append(X)
            y_data_list.append(y)

    return X_data_list, y_data_list, dataset_name


def sample_large_datasets(X_data, y_data):
    """
    For large datasets with over 10000 samples, resample the data to only include
    10000 random samples.
    """
    inds = [i for i in range(X_data.shape[0])]
    fin = sorted(sample(inds, 10000))
    return X_data[fin], y_data[fin]


# Load data from CC18 data set suite

if To_load:
    X_data_list, y_data_list, dataset_name = load_cc18()


# Generate all combinations of nodes to tune over
test_list = [20, 100, 180, 260, 340, 400]

# [NM] - Change
#test_list = [340, 400] #, 100, 180, 260, ]
two_layer = list(itertools.combinations(test_list, 2))
three_layer = list(itertools.combinations(test_list, 3))

node_range = test_list + two_layer + three_layer


# Empty list to record optimal parameters
all_parameters = []


# Choose dataset indices
dataset_indices = [i for i in range(num_datasets)]


# For each dataset, use randomized hyperparameter search to optimize parameters
# [NM] change
#for dataset_index, dataset in enumerate(dataset_indices):
for dataset_index, dataset in enumerate(range(num_datasets)):
    dict_best_params={}
    
    print("\n\nCurrent Dataset: ", dataset)

    X = X_data_list[dataset]
    y = y_data_list[dataset]

    # If data set has over 10000 samples, resample to contain 10000
    if X.shape[0] > 10000:
        X, y = sample_large_datasets(X, y)

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    if algorithms_to_rerun['DN']:
        # Deep network hyperparameters
        parameters = {"hidden_layer_sizes": node_range, "alpha": [0.0001, 0.001, 0.01, 0.1]}   
        
        # [NM]
    
        
         #[NM] - Change
        mlp = MLPClassifier(max_iter=200)
        clf = RandomizedSearchCV(mlp, parameters, n_jobs=-1, cv=None, verbose=1)#GFGHG
        clf.fit(X, y)
        best_params = clf.best_params_
        dict_best_params['DN']=best_params
        allparams = clf.cv_results_["params"]
        
    # Random forest hyperparameters
    if algorithms_to_rerun['RF']:
        p = X.shape[1]    
        l = list(
            set([round(p / 4), round(np.sqrt(p)), round(p / 3), round(p / 1.5), round(p)])
        )
        parameters_rf = {"max_features": l}
        rf = RandomForestClassifier(n_estimators=500)
        clfrf = RandomizedSearchCV(rf, parameters_rf, n_jobs=-1, verbose=1)
        clfrf.fit(X, y)
        best_paramsrf = clfrf.best_params_
        dict_best_params['RF']=best_paramsrf
        allparamsrf = clfrf.cv_results_["params"]
    # [NM]
    if algorithms_to_rerun['GBDT']:
        parameters_GBDT={'max_depth':[1,2,3]} #'learning_rate':1.0, 
        GBDT = GradientBoostingClassifier(n_estimators=500)
        clfGBDT = RandomizedSearchCV(GBDT, parameters_GBDT, n_jobs=-1, verbose=1)
        clfGBDT.fit(X, y)
        #
    #    
    #    
        allparamsgbdt = clfGBDT.cv_results_["params"]           
        best_paramsGBDT = clfGBDT.best_params_
        dict_best_params['GBDT']=best_paramsGBDT
        
    #all_parameters.append([best_params, best_paramsrf,best_paramsGBDT])
    all_parameters.append(dict_best_params.values())

# [NM] - Change


# Save optimal parameters to txt file
file_exists = os.path.exists("metrics/cc18_all_parameters.txt")

if to_use_prev_parameters and file_exists: # To add GBDT to existing file
    all_params = read_params_txt("metrics/cc18_all_parameters.txt")
    with open("metrics/cc18_all_parameters.txt", "w") as f:
        for c_item,item in enumrate(all_parameters):
            item_prev=all_params[c_item]
            if isinstance(item,dict):
                item_prev.append(item)
            else:
                isinstance(item,dict)
                item_prev.append(item[-1])
            f.write("%s\n" % item_prev)


else:
    with open("metrics/cc18_all_parameters.txt", "w") as f:
        for item in all_parameters:
            f.write("%s\n" % item)
        
