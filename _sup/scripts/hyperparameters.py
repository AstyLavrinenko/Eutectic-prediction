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
    study.optimize(objective, n_trials=100)
    trial = study.best_trial
    results.loc[ML_name, 'best_params'] = trial.params
    results.loc[ML_name, 'rmse'] = trial.value
    visualization.matplotlib.plot_optimization_history(study)
    visualization.matplotlib.plot_param_importances(study)

#%%
y = df['T_EP']
x = df[features]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=df['bins'])
bins = x_train.bins
x_train = x_train.drop(['bins'],axis=1)
x_train_new = MinMaxScaler().fit_transform(x_train)
#learning_curve(estimator, X, y, *, groups=None, train_sizes=array([0.1, 0.33, 0.55, 0.78, 1.]), cv=None, scoring=None, exploit_incremental_learning=False, n_jobs=None, pre_dispatch='all', verbose=0, shuffle=False, random_state=None, error_score=nan, return_times=False, fit_params=None)

model = RandomForestRegressor(random_state = 42)
train_sizes, train_scores, test_scores = learning_curve(
        model,
        x_train_new,
        y_train,
        scoring='r2',
        cv=KFold(n_splits = 5, shuffle = True, random_state = 42),
        n_jobs=-1
    )

plt.figure(figsize=(8,5))
plt.errorbar(train_sizes, [np.mean(test_scores[idx]) for idx in range(5)], yerr=[np.std(test_scores[idx]) for idx in range(5)], ecolor='lightgreen', capsize=5, marker = 'o', markersize = 8, color='green', linewidth = 2, label='Cross-validation score')
plt.errorbar(train_sizes, [np.mean(train_scores[idx]) for idx in range(5)], yerr=[np.std(train_scores[idx]) for idx in range(5)], ecolor='lightblue', capsize=5, marker = 'o', markersize = 8, color='blue', linewidth = 2, label='Training score')
plt.xlabel('Size of train dataset', {'fontsize': 14})
plt.xticks(fontsize = 14)
plt.ylabel('Score (RMSE)', {'fontsize': 14})
plt.yticks([0.7,0.75,0.8,0.85,0.9,0.95,1], fontsize = 14)
plt.legend(fontsize = 14, loc='lower right')
plt.show()

params_opt = {'learning_rate':0.4, 
              'n_estimators':200, 
              'min_samples_split':0.2, 
              'min_samples_leaf':0.1, 
              'max_depth':4,
              'max_features':12, 
              'loss':'squared_error', 
              'random_state':42}
model_opt = RandomForestRegressor(**params_opt)

train_sizes, train_scores, test_scores = learning_curve(
        model_opt,
        x_train_new,
        y_train,
        scoring='r2',
        cv=KFold(n_splits = 5, shuffle = True, random_state = 42),
        n_jobs=-1
    )

plt.figure(figsize=(8,5))
plt.errorbar(train_sizes, [np.mean(test_scores[idx]) for idx in range(5)], yerr=[np.std(test_scores[idx]) for idx in range(5)], ecolor='lightgreen', capsize=5, marker = 'o', markersize = 8, color='green', linewidth = 2, label='Cross-validation score')
plt.errorbar(train_sizes, [np.mean(train_scores[idx]) for idx in range(5)], yerr=[np.std(train_scores[idx]) for idx in range(5)], ecolor='lightblue', capsize=5, marker = 'o', markersize = 8, color='blue', linewidth = 2, label='Training score')
plt.xlabel('Size of train dataset', {'fontsize': 14})
plt.xticks(fontsize = 14)
plt.ylabel('Score (RMSE)', {'fontsize': 14})
plt.yticks([0.7,0.75,0.8,0.85,0.9,0.95,1], fontsize = 14)
plt.legend(fontsize = 14, loc='lower right')
plt.show()

#%%
y = df['T_EP']
x = df[features]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=df['bins'])
bins = x_train.bins
x_train = x_train.drop(['bins'],axis=1)
x_test = x_test.drop(['bins'],axis=1)

model = RandomForestRegressor(random_state = 42)
scaler = MinMaxScaler()
x_train_new = scaler.fit_transform(x_train)
x_test_new = scaler.transform(x_test)

model.fit(x_train_new, y_train)
r2,mae,rmse = fit_evaluate.evaluate_model(model,x_test_new,y_test)
print('R2: ',r2,' MAE: ',mae,' RMSE ',rmse)

params_opt = {'learning_rate':0.4, 
              'n_estimators':200, 
              'min_samples_split':0.2, 
              'min_samples_leaf':0.1, 
              'max_depth':4,
              'max_features':12, 
              'loss':'squared_error', 
              'random_state':42}
model_opt = RandomForestRegressor(**params_opt)
scaler = MinMaxScaler()
x_train_new = scaler.fit_transform(x_train)
x_test_new = scaler.transform(x_test)

model_opt.fit(x_train_new, y_train)
r2_opt,mae_opt,rmse_opt = fit_evaluate.evaluate_model(model_opt,x_test_new,y_test)
print('R2: ',r2_opt,' MAE: ',mae_opt,' RMSE: ',rmse_opt)