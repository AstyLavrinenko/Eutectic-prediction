'''
Feature selection
'''
#%% Imports
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, RFE, SelectPercentile, f_regression

#%% Functions
def split_by_system(df, test_size, to_drop):
    data_train_new = pd.DataFrame()
    data_test_new = pd.DataFrame()
    data = df.drop_duplicates(subset = ['Smiles_1', 'Smiles_2'], keep = 'first').reset_index(drop = True)
    data_train, data_test = train_test_split(data, test_size = test_size, random_state = 42)
    for idx in data_train.index:
        smiles_1 = data_train.loc[idx, 'Smiles_1']
        smiles_2 = data_train.loc[idx, 'Smiles_2']
        data_train_new = data_train_new.append(df.iloc[df[(df['Smiles_1'] == smiles_1) & (df['Smiles_2'] == smiles_2)].index])
    for idx in data_test.index:
        smiles_1 = data_test.loc[idx, 'Smiles_1']
        smiles_2 = data_test.loc[idx, 'Smiles_2']
        data_test_new = data_test_new.append(df.iloc[df[(df['Smiles_1'] == smiles_1) & (df['Smiles_2'] == smiles_2)].index])
    x_train = data_train_new.drop(to_drop, axis = 1)
    x_test = data_test_new.drop(to_drop, axis = 1)
    y_train = data_train_new['T_EP']
    y_test = data_test_new['T_EP']
    return x_train, x_test, y_train, y_test

def fit_selector(f_selector, x_train, y_train):
    features = x_train.columns.tolist()
    results = pd.DataFrame({'features': features})
    kfold = KFold(n_splits = 10, shuffle = True, random_state = 42)
    k = 0
    for train_idx, val_idx in kfold.split(x_train):
        k += 1
        x_train_new, x_val = np.array(x_train)[train_idx], np.array(x_train)[val_idx]
        y_train_new, y_val = y_train[train_idx], y_train[val_idx]
        
        scaler_x = StandardScaler()
        x_train_new = scaler_x.fit_transform(x_train_new)
        x_val = scaler_x.transform(x_val)
        scaler_y = StandardScaler()        
        y_train_new = np.ravel(scaler_y.fit_transform(y_train_new))
        
        selector = f_selector
        selector.fit(x_train_new, y_train_new)
        results[f'fold_{k}'] = selector.get_support()
    results['sum'] = np.sum(results, axis = 1)
    threshold = np.mean(results['sum'])
    for idx in results.index:
        if results.loc[idx, 'sum'] >= threshold:
            results.loc[idx, 'final'] = 1
        else:
            results.loc[idx, 'final'] = 0
    return(results)

#%% Read of data
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

#%% Feature selection
selector = VarianceThreshold()
selector.fit_transform(x_train)
features = x_train.loc[:, selector.get_support()].columns.tolist()

x_train = x_train[features]

selector = RFE(RandomForestRegressor(random_state = 42), n_features_to_select = 14, step = 1)
results = fit_selector(selector, x_train, y_train)
features_rfe = []
for idx in results.index:
    if results.loc[idx, 'final'] == 1:
        features_rfe.append(results.loc[idx, 'features'])
        
x_train = x_train[features_rfe]

selectors = {'from_RF': SelectFromModel(estimator = RandomForestRegressor(random_state = 42)),
           'pearson': SelectPercentile(f_regression, percentile = 5)}

selected = pd.DataFrame({'features': x_train.columns.tolist()})
features_new = []
for name, func in selectors.items():
    results = fit_selector(func, x_train, y_train)
    selected[f'{name}'] = results['final']
selected['sum'] = selected.sum(axis = 1, numeric_only=True)
selected = selected.sort_values(by = ['sum'], ascending=False)
threshold = np.mean(selected['sum'])
for idx in selected.index:
    if selected.loc[idx, 'sum'] >= threshold:
        features_new.append(selected.loc[idx, 'features'])
