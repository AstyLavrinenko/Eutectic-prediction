''' Hyperparameters '''

import add_descriptors
import fit_functions

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import GridSearchCV

#%% Read data and add descriptors
df = pd.read_csv('../descriptors/mixture/main.csv')
df = add_descriptors.descriptors().add_selected_features(df)
to_drop = ['Component#1','Smiles#1','Component#2','Smiles#2','T_EP','PD']
df = pd.get_dummies(df, columns = ['Type'])

y = df['T_EP']
x = df.drop(to_drop, axis = 1)
cv = fit_functions.custom_cv(x, y, df.groups, 1, 0.2)
for train_idx, val_idx in cv:
    x_train = x.iloc[train_idx]
    y_train = y.iloc[train_idx]
groups = x_train.groups
x_train = x_train.drop(['groups'],axis=1)
results = pd.DataFrame()

#%% Hyperparameters for GBR
n_estimators = np.arange(10, 80, 5)
max_features = np.arange(3, 14, 1)
max_depth = np.arange(3, 10, 1)
min_samples_split = np.arange(0.05,0.45,0.05)
min_samples_leaf = np.arange(0.02,0.21,0.02)
learning_rate = np.arange(0.1,0.45,0.05)
params_GBR = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'learning_rate': learning_rate}

#%% Hyperparameters for RFR
n_estimators = np.arange(10, 80, 5)
max_features = np.arange(3, 14, 1)
max_depth = np.arange(3, 10, 1)
min_samples_split = np.arange(0.05,0.45,0.05)
min_samples_leaf = np.arange(0.02,0.21,0.02)

params_RFR = {'n_estimators': n_estimators,
          'max_features': max_features,
          'max_depth': max_depth,
          'min_samples_split': min_samples_split,
          'min_samples_leaf': min_samples_leaf}

#%% Hyperparameters for KNN
n_neighbors = np.arange(5, 20, 1)
metric = ['minkowski','euclidean','manhattan','hamming']
p = np.arange(0.1,2.6,0.1)

params_KNN = {'n_neighbors' : n_neighbors,
                'metric' : metric
                }

#%% Hyperparameters for SVR
C = np.arange(0.5, 6.5, 0.5)
gamma = np.arange(0, 5, 0.5)
kernel = ['rbf', 'poly', 'sigmoid']

params_SVR = {'C': C, 
              'gamma': gamma,
              'kernel': kernel}

#%% Grid search
models={'GBR':(GradientBoostingRegressor(),params_GBR),
        'RFR':(RandomForestRegressor(n_jobs=-1),params_RFR),
        'KNN':(KNeighborsRegressor(),params_KNN),
        'SVR':(SVR(),params_SVR)}

for ML_name,ML_params in models.items():
    y_train_new = np.array(y_train).reshape(-1, 1)
    scaler = MinMaxScaler()
    x_train_new = scaler.fit_transform(x_train)
    model = ML_params[0]
    params = ML_params[1]
    if ML_name == 'SVR':
        scaler_y = StandardScaler()
        y_train_new = np.ravel(scaler_y.fit_transform(y_train_new))
    model_gs = GridSearchCV(estimator=model,param_grid=params,scoring=['r2','neg_mean_squared_error'],refit='neg_mean_squared_error',cv=fit_functions.custom_cv(x_train_new, y_train_new, groups, 5, 0.2),n_jobs=-1,verbose=2)
    model_gs.fit(x_train_new, np.ravel(y_train_new))
    results.loc[f'{ML_name}', 'best_params'] = str(model_gs.best_estimator_) 
results.to_csv('../results/grid_search.csv')
