"""
Author: Michael Ainsworth
"""

#%% Imports

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
#%% Define executrion variables (to save memory & execution time)
reload_data = False # indicator of whether to upload the data again
nodes_combination = [20, 100, 180, 260, 340, 400] # default is [20, 100, 180, 260, 340, 400]
dataset_indices_max=72 
max_shape_to_run=10000
alpha_range_nn=[0.0001, 0.001, 0.01, 0.1]
subsample=[0.5,0.8,1.0]


#models
models_to_run={'RF':0,'DN':0,'GBDT':1} # Define which models to run
classifiers={'DN':MLPClassifier(max_iter=200), 'RF':RandomForestClassifier(n_estimators=500), 'GBDT': GradientBoostingClassifier(n_estimators=500)}
vararginCV={'DN':{'n_jobs':-1,'verbose':1,'cv':None},
            'RF':{'n_jobs':-1,'verbose':1,'cv':None},
            'GBDT':{'n_jobs':None,'verbose':1,'cv':None}}
#%% function to return to default values
def return_to_default():
    nodes_combination = [20, 100, 180, 260, 340, 400]
    dataset_indices_max = 72 
    max_shape_to_run = 10000
    alpha_range_nn = [0.0001, 0.001, 0.01, 0.1]
    models_to_run={'RF':1,'DN':1,'GBDT':1}
    return nodes_combination,dataset_indices_max,max_shape_to_run

#%% function so save cc18_all_parameters file

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


"""
Organize the data
"""

# Load data from CC18 data set suite
if (reload_data or 'task_id' not in locals()): # Load the data only if required (by reload_data or if it is not defined)
    X_data_list, y_data_list, dataset_name = load_cc18()


# Empty dict to record optimal parameters
all_parameters = {model_name:None for model_name,val in models_to_run if val==1}
best_parameters = {model_name:None for model_name,val in models_to_run if val==1}


# Choose dataset indices
dataset_indices = [i for i in range(dataset_indices_max)]


"""
Deep Neural Network
"""
# Generate all combinations of nodes to tune over
test_list = nodes_combination;
two_layer = list(itertools.combinations(test_list, 2))
three_layer = list(itertools.combinations(test_list, 3))

node_range = test_list + two_layer + three_layer

# Functions to calculate model performance and parameters.

def create_parameters(model_name,varargin):
    if model_name=='DN':
        parameters={"hidden_layer_sizes": varargin['node_range'], "alpha": varargin['alpha_range_nn']}
    elif model_name=='RF':
        parameters={"max_features": list(set([round(p / 4), round(np.sqrt(p)), round(p / 3), round(p / 1.5), round(p)]))}
    elif model_name=='GBDT':
        parameters={'learning_rate':varargin['alpha_range_nn'],'subsample':varargin['subsample']}
    else:
        raise ValueError("Model name is invalid. Please check the keys of models_to_run")
    return parameters
        
def do_calcs_per_model(all_parameters,best_parameters, allparams,model_name,varargin,varCV,classifiers,X,y):
    model=classifiers[model_name]
    varCVmodel=varCV[model_name]
    parameters=create_parameters(model_name,varargin)
    clf = RandomizedSearchCV(model, parameters, n_jobs=varCVmodel[n_jobs], cv=varCVmodel[cv], verbose=varCVmodel[verbose])
    clf.fit(X, y)
    all_parameters[model_name]=parameters
    best__parameters[model_name]=clf.best_params_
    allparams[model_name]=clf.cv_results_["params"]
    return all_parameters, best_parameters,all_params


# For each dataset, use randomized hyperparameter search to optimize parameters
for dataset_index, dataset in enumerate(dataset_indices):

    print("\n\nCurrent Dataset: ", dataset)

    X = X_data_list[dataset]
    y = y_data_list[dataset]

    # If data set has over 10000 samples, resample to contain 10000
    if X.shape[0] > max_shape_to_run:
        X, y = sample_large_datasets(X, y)
        
    # Standart Scaler
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    
    p = X.shape[1]
    create_parameters(models_to_run)
    for model_name,val_run in models_to_run.items():
        if val==1:
            if model_name not in classifires:
                raise ValueError('Model name is not defined in the classifiers dictionary')
            else:
                all_parameters, best_parameters,all_params=do_calcs_per_model(all_parameters,
                                                                              best_parameters, 
                                                                              allparams,
                                                                              model_name,
                                                                              varargin,vararginCV,classifiers,X,y)





# Save optimal parameters to txt file
with open("metrics/cc18_all_parameters.txt", "w") as f:
    for item in all_parameters:
        f.write("%s\n" % item)