''' Comparison of ML models '''

import fit_functions
import add_descriptors

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

#%% Read data
df = pd.read_csv('../descriptors/mixture/main.csv')
df = add_descriptors.descriptors().add_selected_features(df)
to_drop = ['Component#1','Smiles#1','Component#2','Smiles#2','T_EP','PD']
df = pd.get_dummies(df, columns = ['Type'])
y = df['T_EP']
x = df.drop(to_drop, axis = 1)
cv = fit_functions.custom_cv(x, y, df.groups, 1, 0.2)
for train_idx, val_idx in cv:
    x_train,x_val = x.iloc[train_idx], x.iloc[val_idx]
    y_train,y_val = y.iloc[train_idx], y.iloc[val_idx]
groups = x_train.groups
x_train = x_train.drop(['groups'],axis=1)

#%% Training of ML models
models = {'RFR': RandomForestRegressor(bootstrap=True, max_depth=4, max_features=8,
                      min_samples_leaf=10, min_samples_split=20,n_estimators=65, n_jobs=-1), 
          'GBR': GradientBoostingRegressor(learning_rate=0.1, max_depth=3, max_features=9,
                          min_samples_leaf=45, min_samples_split=30,n_estimators=80), 
          'KNN': KNeighborsRegressor(n_neighbors=10,weights='uniform',metric='manhattan'), 
          'SVR': SVR(C=9, gamma=0.35,kernel='rbf'),
          'MLR': LinearRegression()
          }

results = pd.DataFrame()
idx=-1
for name, model in models.items():
    idx+=1
    params = {'ML_model':model, 
              'ML_name':name, 
              'x_train':np.array(x_train), 
              'groups': groups,
              'y_train':np.array(y_train).reshape(-1, 1), 
              'scaler_y':False, 
              'scaler_x':MinMaxScaler(), 
              'n_kfold':5,
              'validation':True,
              'x_val':x_val.drop(['groups'],axis=1),
              'y_val':y_val}
    if name == 'SVR':
        params['scaler_y'] = True
    result_dict = fit_functions.fit_model_kfold(**params)
    for key,value in result_dict.items():
        results.loc[idx,key] = str(value)
results = results.sort_values(by=['rmse_val'])
results.to_csv('../results/ML_model_selection.csv', index = False)
