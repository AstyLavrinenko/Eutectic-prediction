'''
Hyperparameters optimization
'''
#%% Imports
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold

from sklearn.ensemble import RandomForestRegressor

#%% Functions
def evaluate_model(model, x, y, scaler_y):
    y_pred = model.predict(x)
    y_pred = scaler_y.inverse_transform(y_pred)
    
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared = False)
    return r2, mae, rmse

def fit_model(ML_model, x_train, y_train):
    cv_results = pd.DataFrame()
    kfold = KFold(n_splits = 10, shuffle = True, random_state = 42)
    for train_idx, val_idx in kfold.split(x_train):
        model = ML_model
        
        x_train_new, x_val = np.array(x_train)[train_idx], np.array(x_train)[val_idx]
        y_train_new, y_val = y_train[train_idx], y_train[val_idx]
        
        scaler_x = StandardScaler()
        x_train_new = scaler_x.fit_transform(x_train_new)
        x_val = scaler_x.transform(x_val)
        scaler_y = StandardScaler()        
        y_train_new = np.ravel(scaler_y.fit_transform(y_train_new))
        
        model.fit(x_train_new, y_train_new)
        
        y_train_new = scaler_y.inverse_transform(y_train_new)
        
        r2_train, mae_train, rmse_train = evaluate_model(model, x_train_new, y_train_new, scaler_y)
        r2_val, mae_val, rmse_val = evaluate_model(model, x_val, y_val, scaler_y)
        
        cv = {
        'r2_train': r2_train,
        'mae_train': mae_train,
        'rmse_train': rmse_train,
        'r2_val': r2_val,
        'mae_val': mae_val,
        'rmse_val': rmse_val}
        cv_results = cv_results.append(cv, ignore_index = True)
        
    cv_results = cv_results.agg(['mean', 'std'])
    result_dict = {}
    for col in cv_results.columns:
        for idx in cv_results.index:
            result_dict[f'{col}_{idx}'] = cv_results.loc[idx, col]
           
    return result_dict

def learning_curve(ML_model, x_train, y_train):
    cv_results = pd.DataFrame()
    kfold = KFold(n_splits = 10, shuffle = True, random_state = 42)
    for train_idx, val_idx in kfold.split(x_train):
        k = 9*len(x_train) // 100
        h = k
        for i in range(10):
            model = ML_model
            x_train_new, x_val = np.array(x_train)[train_idx[:k]], np.array(x_train)[val_idx]
            y_train_new, y_val = y_train[train_idx[:k]], y_train[val_idx]
        
            scaler_x = StandardScaler()
            x_train_new = scaler_x.fit_transform(x_train_new)
            x_val = scaler_x.transform(x_val)
            scaler_y = StandardScaler()        
            y_train_new = np.ravel(scaler_y.fit_transform(y_train_new))
        
            model.fit(x_train_new, y_train_new)
        
            y_train_new = scaler_y.inverse_transform(y_train_new)
        
            r2_train, mae_train, rmse_train = evaluate_model(model, x_train_new, y_train_new, scaler_y)
            r2_val, mae_val, rmse_val = evaluate_model(model, x_val, y_val, scaler_y)
        
            cv = {'train_size': k,
                'r2_train': r2_train,
                'mae_train': mae_train,
                'rmse_train': rmse_train,
                'r2_val': r2_val,
                'mae_val': mae_val,
                'rmse_val': rmse_val}
            cv_results = cv_results.append(cv, ignore_index = True)
            
            k += h
            if i == 8:
                k = len(train_idx)
        
    cv_results = cv_results.groupby(['train_size'], as_index=False).agg(['mean', 'std']).reset_index()
    result_dict = {}
    for idx in cv_results.index:
        name = cv_results.loc[idx, 'train_size']['']
        result_dict[name] = {}
        for col, sub_col in cv_results.columns:
            if col == 'train_size':
                continue
            result_dict[name][f'{col}_{sub_col}'] = cv_results.loc[idx, col][sub_col]
           
    return result_dict
#%% Read data
df = pd.DataFrame()
for csv in os.listdir('descriptors'):
    addend = pd.read_csv(f'descriptors/{csv}')
    df = pd.concat([df, addend], axis = 1)
df = df.loc[:,~df.columns.duplicated()]

to_drop = ['Component_1', 'Smiles_1', 'Component_2', 'Smiles_2', 'T_EP']
features_1 = ['T_1', 'X_1', 'T_2']
features_2 = ['T_1', 'mu_1', 'H_MF_1', 'X_1', 'T_2', 'mu_2', 'H_int_2', 'mu_gas_2', 'MolLogP_2', 'WAPS_2', 'area_p_2']

features = features_1
#features = features_2
model_param = 'model_1'
#model_param = 'model_2'
df = df[to_drop + features]
df = df.dropna()

y = df['T_EP']
x = df.drop(to_drop, axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
x_train_init = x_train
y_train = np.array(y_train).reshape(-1, 1)

#%% Variations of hyperparameters
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num = 19)]
max_depth = [int(x) for x in np.linspace(1, 30, num = 16)] + [None]
min_samples_split = [int(x) for x in np.linspace(2, 20, num = 10)]
min_samples_leaf = [int(x) for x in np.linspace(1, 20, num = 11)]
params = {'n_estimators': n_estimators,
          'max_depth': max_depth,
          'min_samples_split': min_samples_split,
          'min_samples_leaf': min_samples_leaf}

#%% n_estimators
results = pd.DataFrame()
for estimator in n_estimators:
    model = RandomForestRegressor(n_estimators=estimator, random_state = 42)
    cv = fit_model(model, x_train, y_train)
    cv['n_estimators'] = estimator
    results = results.append(cv, ignore_index = True)

plt.clf()
plt.figure(figsize=(8,5))
plt.errorbar(results['n_estimators'], results['r2_val_mean'], yerr=results['r2_val_std'], ecolor='lightgreen', capsize=5, marker = 'o', markersize = 8, color='green', linewidth = 2, label='Cross-validation score')
plt.errorbar(results['n_estimators'], results['r2_train_mean'], yerr=results['r2_train_std'], ecolor='lightblue', capsize=5, marker = 'o', markersize = 8, color='blue', linewidth = 2, label='Training score')
plt.xlabel('n_estimators', {'fontsize': 14})
plt.xticks(fontsize = 14)
plt.ylabel('Score (R2)', {'fontsize': 14})
plt.yticks(fontsize = 14)
plt.legend(fontsize = 14, loc='lower right')
plt.savefig(f'hyperparam/{model_param}/n_estimators.png')
plt.show()
     
#%% max_depth
results = pd.DataFrame()
estimator_label = []
for estimator in max_depth:
    model = RandomForestRegressor(max_depth=estimator, random_state=42)
    cv = fit_model(model, x_train, y_train)
    cv['max_depth'] = estimator
    if estimator == None:
        cv['max_depth'] = 32
    results = results.append(cv, ignore_index = True)

plt.clf()
plt.figure(figsize=(8,5))
plt.errorbar(results['max_depth'], results['r2_val_mean'], yerr=results['r2_val_std'], ecolor='lightgreen', capsize=5, marker = 'o', markersize = 8, color='green', linewidth = 2, label='Cross-validation score')
plt.errorbar(results['max_depth'], results['r2_train_mean'], yerr=results['r2_train_std'], ecolor='lightblue', capsize=5, marker = 'o', markersize = 8, color='blue', linewidth = 2, label='Training score')
plt.xlabel('max_depth', {'fontsize': 14})
plt.xticks([2, 6, 10, 14, 18, 22, 26, 30, 32], [2, 6, 10, 14, 18, 22, 26, 30, 'None'], fontsize = 14)
plt.ylabel('Score (R2)', {'fontsize': 14})
plt.yticks(fontsize = 14)
plt.legend(fontsize = 14, loc='lower right')
plt.savefig(f'hyperparam/{model_param}/max_depth.png')
plt.show()
        
#%% min_samples_split
results = pd.DataFrame()
for estimator in min_samples_split:
    model = RandomForestRegressor(min_samples_split=estimator, random_state=42)
    cv = fit_model(model, x_train, y_train)
    cv['min_samples_split'] = estimator
    results = results.append(cv, ignore_index = True)

plt.clf()
plt.figure(figsize=(8,5))
plt.errorbar(results['min_samples_split'], results['r2_val_mean'], yerr=results['r2_val_std'], ecolor='lightgreen', capsize=5, marker = 'o', markersize = 8, color='green', linewidth = 2, label='Cross-validation score')
plt.errorbar(results['min_samples_split'], results['r2_train_mean'], yerr=results['r2_train_std'], ecolor='lightblue', capsize=5, marker = 'o', markersize = 8, color='blue', linewidth = 2, label='Training score')
plt.xlabel('min_samples_split', {'fontsize': 14})
plt.xticks(fontsize = 14)
plt.ylabel('Score (R2)', {'fontsize': 14})
plt.yticks(fontsize = 14)
plt.legend(fontsize = 14, loc='lower right')
plt.savefig(f'hyperparam/{model_param}/min_samples_split.png')
plt.show()

#%% min_samples_leaf
results = pd.DataFrame()
for estimator in min_samples_leaf:
    model = RandomForestRegressor(min_samples_leaf=estimator, random_state=42)
    cv = fit_model(model, x_train, y_train)
    cv['min_samples_leaf'] = estimator
    results = results.append(cv, ignore_index = True)

plt.clf()
plt.figure(figsize=(8,5))
plt.errorbar(results['min_samples_leaf'], results['r2_val_mean'], yerr=results['r2_val_std'], ecolor='lightgreen', capsize=5, marker = 'o', markersize = 8, color='green', linewidth = 2, label='Cross-validation score')
plt.errorbar(results['min_samples_leaf'], results['r2_train_mean'], yerr=results['r2_train_std'], ecolor='lightblue', capsize=5, marker = 'o', markersize = 8, color='blue', linewidth = 2, label='Training score')
plt.xlabel('min_samples_leaf', {'fontsize': 14})
plt.xticks(fontsize = 14)
plt.ylabel('Score (R2)', {'fontsize': 14})
plt.yticks(fontsize = 14)
plt.legend(fontsize = 14, loc='lower right')
plt.savefig(f'hyperparam/{model_param}/min_samples_leaf.png')
plt.show()
