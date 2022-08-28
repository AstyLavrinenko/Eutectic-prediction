''' Comparison of classic models '''
#%% Imports
import fit_evaluate
import add_descriptors

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#%% Read data
df = pd.read_csv('../descriptors/mixture/main.csv')
df_descriptors = add_descriptors.descriptors().add_several(df)
df['ln_x#1'] = np.log(df['X#1'])
df['ln_x#2'] = np.log(1-df['X#1'])
df['frac'] = df['X#1']/(1-df['X#1'])
df['T_frac'] = df['X#1']*df['T#1']+(1-df['X#1'])*df['T#2']
df['T_frac#1'] = df['X#1']*df['T#1']
df['T_frac#2'] = (1-df['X#1'])*df['T#2']

df_descriptors = df_descriptors.dropna(axis=1)
to_drop = ['Component#1', 'Smiles#1', 'Component#2', 'Smiles#2', 'T_EP']

train_idx, test_idx = fit_evaluate.split_by_bins(df, 0.2)
y = df['T_EP']
x = df.drop(to_drop, axis = 1)
x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

#%%Training of classic models
models = {'RFR': RandomForestRegressor(), 'GBR': GradientBoostingRegressor(), 
          'KNN': KNeighborsRegressor(), 'SVR': SVR(), 'MLR': LinearRegression()}

results = pd.DataFrame()
for name, model in models.items():
    params = {'ML_model':model, 
              'ML_name':name, 
              'x_train':np.array(x_train), 
              'y_train':np.array(y_train).reshape(-1, 1), 
              'scaler_y':False, 
              'scaler_x':MinMaxScaler(), 
              'n_kfold':10}
    if name == 'SVR':
        params['scaler_y'] = StandardScaler()
    result_dict = fit_evaluate.fit_model_kfold(**params)
    results = results.append(result_dict, ignore_index = True)
results = results.sort_values(by=['rmse_val_mean'])
results.to_csv('../results/ML_model_selection.csv', index = False)
