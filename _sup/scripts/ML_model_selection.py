''' Comparison of classic models '''
#%% Imports
import fit_evaluate
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
df = df.dropna(axis=1)
to_drop = ['Component#1','Smiles#1','Component#2','Smiles#2','T_EP','PD']
df = pd.get_dummies(df, columns = ['Type'])
y = df['T_EP']
x = df.drop(to_drop, axis = 1)
cv = fit_evaluate.custom_cv(x, y, df.groups, 1, 0.2)
for train_idx, test_idx in cv:
    x_train,x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train,y_test = y.iloc[train_idx], y.iloc[test_idx]
groups = x_train.groups
x_train = x_train.drop(['groups'],axis=1)

#%%Training of classic models
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
importances = []
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
              'x_val':x_test.drop(['groups'],axis=1),
              'y_val':y_test}
    if name == 'SVR':
        params['scaler_y'] = True
    result_dict, importance = fit_evaluate.fit_model_kfold(**params)
    importances.append(importance)
    for key,value in result_dict.items():
        results.loc[idx,key] = str(value)
results = results.sort_values(by=['rmse_val'])
results.to_csv('../results/ML_model_selection.csv', index = False)
df_importance = pd.DataFrame(index=x_train.columns)
for i,key in enumerate(models.keys()):
    df_importance_new=pd.DataFrame(data=np.transpose(importances[i]),index=x_train.columns)
    df_importance_new['mean']=df_importance_new.mean(axis=1)
    df_importance[f'{key}'] = df_importance_new['mean']
