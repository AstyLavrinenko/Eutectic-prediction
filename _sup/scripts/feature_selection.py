'''
Feature selection
'''
#%% Imports
import fit_evaluate
import add_descriptors

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, RFE, SelectPercentile, f_regression

#%% Functions
def fit_selector(f_selector, x_train, y_train):
    features = x_train.columns.tolist()
    results = pd.DataFrame({'features': features})
    kfold = KFold(n_splits = 10, shuffle = True, random_state = 42)
    k = 0
    for train_idx, val_idx in kfold.split(x_train):
        k += 1
        x_train_new, x_val = np.array(x_train)[train_idx], np.array(x_train)[val_idx]
        y_train_new, y_val = np.array(y_train)[train_idx], np.array(y_train)[val_idx]
        
        scaler_x = StandardScaler()
        x_train_new = scaler_x.fit_transform(x_train_new)
        x_val = scaler_x.transform(x_val)
        
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
df=pd.read_csv('../descriptors/mixture/main.csv')
df=add_descriptors.descriptors().add_several(df)
df['ln_x#1'] = np.log(df['X#1'])
df['ln_x#2'] = np.log(1-df['X#1'])
df['frac'] = df['X#1']/(1-df['X#1'])
df['T_frac'] = df['X#1']*df['T#1']+(1-df['X#1'])*df['T#2']
df['T_frac#1'] = df['X#1']*df['T#1']
df['T_frac#2'] = (1-df['X#1'])*df['T#2']
df = df.dropna(axis=1)
to_drop = ['Component#1', 'Smiles#1', 'Component#2', 'Smiles#2', 'T_EP', 'Phase_diagram']

train_idx, test_idx = fit_evaluate.split_by_bins(df, 0.2)
y = df['T_EP']
x = df.drop(to_drop, axis = 1)
x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

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
with open('../results/features.txt', 'w') as file:
    for feat in features_new:
        file.write(feat+'\n')