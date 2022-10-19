''' Hyperparameters '''

import add_descriptors
import fit_evaluate

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, make_scorer, r2_score, mean_absolute_error
import optuna
from optuna import visualization
import matplotlib.pyplot as plt

#%%
def search_RFR(trial):
    y = df['T_EP']
    x = df.drop(to_drop, axis = 1)
    cv = fit_evaluate.custom_cv(x, y, df.groups, 1, 0.2)
    for train_idx, test_idx in cv:
        x_train = x.iloc[train_idx]
        y_train = y.iloc[train_idx]
    groups = x_train.groups
    x_train = x_train.drop(['groups'],axis=1)
    search_space = {'n_estimators': trial.suggest_int('n_estimators',50,300),
                    'min_samples_split': trial.suggest_int('min_samples_split',2,20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf',2,20),
                    'max_depth': trial.suggest_int('max_depth',1,20),
                    'max_features': trial.suggest_int('max_features',1,15),
                    'criterion': trial.suggest_categorical('criterion',['squared_error','absolute_error','poisson']),
                    'random_state':42,
                    'n_jobs':-1}
    regressor = RandomForestRegressor(**search_space)
    
    params = {'ML_model':regressor, 
              'ML_name':'RFR',
              'x_train':np.array(x_train), 
              'groups': groups,
              'y_train':np.array(y_train).reshape(-1, 1), 
              'scaler_y':False,
              'scaler_x':MinMaxScaler(), 
              'n_kfold':5}
    result_dict = fit_evaluate.fit_model_kfold(**params)
    return result_dict['rmse_val_mean']

def search_GBR(trial):
    y = df['T_EP']
    x = df.drop(to_drop, axis = 1)
    cv = fit_evaluate.custom_cv(x, y, df.groups, 1, 0.2)
    for train_idx, test_idx in cv:
        x_train = x.iloc[train_idx]
        y_train = y.iloc[train_idx]
    groups = x_train.groups
    x_train = x_train.drop(['groups'],axis=1)
    search_space = {'learning_rate': trial.suggest_float('learning_rate',0.1,0.5,step=0.05),
                    'n_estimators':trial.suggest_int('n_estimators',50,300), 
                    'min_samples_split':trial.suggest_int('min_samples_split',2,20), 
                    'min_samples_leaf':trial.suggest_int('min_samples_leaf',2,20), 
                    'max_depth':trial.suggest_int('max_depth',1,20),
                    'max_features':trial.suggest_int('max_features',1,15), 
                    'loss':trial.suggest_categorical('loss',['squared_error', 'absolute_error', 'huber', 'quantile']), 
                    'random_state':42}
    regressor = GradientBoostingRegressor(**search_space)
    
    params = {'ML_model':regressor, 
              'ML_name':'GBR',
              'x_train':np.array(x_train), 
              'groups': groups,
              'y_train':np.array(y_train).reshape(-1, 1), 
              'scaler_y':False,
              'scaler_x':MinMaxScaler(), 
              'n_kfold':5}
    result_dict = fit_evaluate.fit_model_kfold(**params)
    return result_dict['rmse_val_mean']

def search_KNN(trial):
    y = df['T_EP']
    x = df.drop(to_drop, axis = 1)
    cv = fit_evaluate.custom_cv(x, y, df.groups, 1, 0.2)
    for train_idx, test_idx in cv:
        x_train = x.iloc[train_idx]
        y_train = y.iloc[train_idx]
    groups = x_train.groups
    x_train = x_train.drop(['groups'],axis=1)
    search_space = {'n_neighbors' : trial.suggest_int('n_neighbors',5,20),
                    'weights' : trial.suggest_categorical('weights',['uniform','distance']),
                    'metric' : trial.suggest_categorical('metric',['minkowski','euclidean','manhattan']),
                    'leaf_size': trial.suggest_int('leaf_size',10,40),
                    'algorithm': trial.suggest_categorical('algorithm',['ball_tree','kd_tree','brute'])}
    regressor = KNeighborsRegressor(**search_space)
    
    params = {'ML_model':regressor, 
              'ML_name':'KNN',
              'x_train':np.array(x_train), 
              'groups': groups,
              'y_train':np.array(y_train).reshape(-1, 1), 
              'scaler_y':False,
              'scaler_x':MinMaxScaler(), 
              'n_kfold':5}
    result_dict = fit_evaluate.fit_model_kfold(**params)
    return result_dict['rmse_val_mean']

def search_SVR(trial):
    y = df['T_EP']
    x = df.drop(to_drop, axis = 1)
    cv = fit_evaluate.custom_cv(x, y, df.groups, 1, 0.2)
    for train_idx, test_idx in cv:
        x_train = x.iloc[train_idx]
        y_train = y.iloc[train_idx]
    groups = x_train.groups
    x_train = x_train.drop(['groups'],axis=1)
    search_space = {'C': trial.suggest_int('C',1,100), 
                    'gamma':trial.suggest_float('gamma',0.001,1),
                    'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])}
    regressor = SVR(**search_space)
    
    params = {'ML_model':regressor, 
              'ML_name':'KNN',
              'x_train':np.array(x_train), 
              'groups': groups,
              'y_train':np.array(y_train).reshape(-1, 1), 
              'scaler_y':StandardScaler(),
              'scaler_x':MinMaxScaler(), 
              'n_kfold':5}
    result_dict = fit_evaluate.fit_model_kfold(**params)
    return result_dict['rmse_val_mean']

#%%
df = pd.read_csv('../descriptors/mixture/main.csv')
df = add_descriptors.descriptors().add_several(df, profile=False)
df=add_descriptors.descriptors().add_profile(df, {add_descriptors.ratio:'all'})
df = df.dropna(axis=1)
to_drop = ['Component#1','Smiles#1','Component#2','Smiles#2','T_EP','Phase_diagram']

#%%
models = {'RFR': search_RFR, 'GBR': search_GBR, 
          'KNN': search_KNN, 'SVR': search_SVR}
results = pd.DataFrame()
for ML_name,objective in models.items():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=200)
    trial = study.best_trial
    results.loc[ML_name, 'best_params'] = str(trial.params)
    results.loc[ML_name, 'rmse'] = trial.value
    visualization.matplotlib.plot_optimization_history(study)
    visualization.matplotlib.plot_param_importances(study)
results.to_csv('../results/hyperparameters.csv')