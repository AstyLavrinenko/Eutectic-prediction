'''
Comparison of classic models
'''
#%% Imports
import os
import pandas as pd
import numpy as np
from time import time

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

#%% Functions
def evaluate_model(model, x, y, scaler_y):
    y_pred = model.predict(x)
    y_pred = scaler_y.inverse_transform(y_pred)    
    
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared = False)
    
    return r2, mae, rmse

def fit_model(ML_model, ML_name, x_train, y_train):
    cv_results = pd.DataFrame()
    kfold = KFold(n_splits = 10, shuffle = True, random_state = 42)
    for train_idx, val_idx in kfold.split(x_train):
        model = ML_model        
        
        x_train_new, x_val = x_train[train_idx], x_train[val_idx]
        y_train_new, y_val = y_train[train_idx], y_train[val_idx]        
        scaler_x = StandardScaler()
        x_train_new = scaler_x.fit_transform(x_train_new)
        x_val = scaler_x.transform(x_val)        
        scaler_y = StandardScaler()        
        y_train_new = np.ravel(scaler_y.fit_transform(y_train_new))
        
        init_time = time()
        model.fit(x_train_new, y_train_new)
        fit_time = time() - init_time
        
        y_train_new = scaler_y.inverse_transform(y_train_new)        
        r2_train, mae_train, rmse_train = evaluate_model(model, x_train_new, y_train_new, scaler_y)
        r2_val, mae_val, rmse_val = evaluate_model(model, x_val, y_val, scaler_y)        
        
        cv = {
        'fit_time': fit_time,
        'r2_train': r2_train,
        'mae_train': mae_train,
        'rmse_train': rmse_train,
        'r2_val': r2_val,
        'mae_val': mae_val,
        'rmse_val': rmse_val}
        cv_results = cv_results.append(cv, ignore_index = True)        
    
    cv_results = cv_results.agg(['mean', 'std'])
    result_dict = {'model_name': ML_name, 'model_params': model.get_params()}
    for col in cv_results.columns:
        for idx in cv_results.index:
            result_dict[f'{col}_{idx}'] = cv_results.loc[idx, col]           
    
    return result_dict
#%% Read data
df = pd.DataFrame()
for csv in os.listdir('descriptors'):
    addend = pd.read_csv(f'descriptors/{csv}')
    df = pd.concat([df, addend], axis = 1)
df = df.loc[:,~df.columns.duplicated()]
df = df.drop(['H_1', 'H_2'], axis = 1)
to_drop = ['Component_1', 'Smiles_1', 'Component_2', 'Smiles_2', 'T_EP']

y = df['T_EP']
x = df.drop(to_drop, axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
x_train_init = x_train
y_train = np.array(y_train).reshape(-1, 1)

#%%Training of classic models
models = {'RFR': RandomForestRegressor(), 'GBR': GradientBoostingRegressor(), 
          'KNN': KNeighborsRegressor(), 'SVR': SVR(), 'MLR': LinearRegression()}

results = pd.DataFrame()
for name, model in models.items():
    result_dict = fit_model(model, name, np.array(x_train), y_train)
    results = results.append(result_dict, ignore_index = True)
results = results.sort_values(by=['rmse_val_mean'])
results.to_csv('ML_model_selection.csv', index = False)
