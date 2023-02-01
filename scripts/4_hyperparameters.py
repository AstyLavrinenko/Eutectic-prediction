''' Hyperparameters '''

import add_descriptors
import fit_functions

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve

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

#%% Hyperparameters for GBR
n_estimators = np.arange(10, 105, 5)
max_features = np.arange(1, 17, 1)
max_depth = np.arange(1, 11, 1)
min_samples_split = np.arange(10, 85, 5)
min_samples_leaf = np.arange(10, 85, 5)
learning_rate = np.arange(0.05, 0.55, 0.05)
params_GBR = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'learning_rate': learning_rate}

#%% Hyperparameters for RFR
n_estimators = np.arange(10, 105, 5)
max_features = np.arange(1, 17, 1)
max_depth = np.arange(1, 11, 1)
min_samples_split = np.arange(10, 85, 5)
min_samples_leaf = np.arange(10, 85, 5)
bootstrap = [True, False]

params_RFR = {'n_estimators': n_estimators,
          'max_features': max_features,
          'max_depth': max_depth,
          'min_samples_split': min_samples_split,
          'min_samples_leaf': min_samples_leaf,
          'bootstrap': bootstrap}

#%% Hyperparameters for KNN
n_neighbors = np.arange(5, 55, 5)
weights = ['uniform','distance']
metric = ['minkowski','euclidean','manhattan','hamming']

params_KNN = {'n_neighbors' : n_neighbors,
                'weights' : weights,
                'metric' : metric}

#%% Hyperparameters for SVR
C = np.arange(0.1, 20.1, 0.1)
gamma = np.arange(0, 0.55, 0.05)
kernel = ['rbf', 'poly', 'sigmoid']

params_SVR = {'C': C, 
              'gamma': gamma,
              'kernel': kernel}

#%% Check the range of hyperparameters

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
    for param_name, param_range in params.items():
        train_score, val_score=validation_curve(estimator=model,X=x_train_new,y=y_train_new,param_name=param_name, param_range=param_range, cv=fit_functions.custom_cv(x_train_new, y_train_new, groups, 5, 0.2), scoring='neg_root_mean_squared_error', n_jobs=-1)
        plt.plot(param_range, np.median(train_score, 1), color='blue', label=f'{ML_name} training score')
        plt.plot(param_range, np.median(val_score, 1), color='red', label=f'{ML_name} validation score')
        plt.legend(loc='best')
        plt.xlabel(f'{param_name}')
        plt.ylabel('RMSE')
        plt.show()