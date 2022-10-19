''' Feature selection '''

import add_descriptors
import fit_functions

import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from minepy import MINE # will work only on Linux
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, SequentialFeatureSelector

#%% Functions        
def apply_MIC(x_train, groups, y_train, features, threshold_MIC, n_splits = 5):
    results = pd.DataFrame({'features': features})
    kfold = fit_functions.custom_cv(x_train,y_train,groups,n_splits,1/n_splits)
    k = 0
    for train_idx, val_idx in kfold:
        k += 1
        x_train_new = np.array(x_train)[train_idx]
        y_train_new = np.array(y_train)[train_idx]
        scaler = MinMaxScaler()
        x_train_new = scaler.fit_transform(x_train_new)
        for idx in results.index:
            selector = MINE(alpha=0.6, c=15, est="mic_approx")
            selector.compute_score(x_train_new[:,idx], y_train_new)
            results.loc[idx, f'MIC_{k}'] = selector.mic()
    results['MIC_sum'] = np.mean(results.drop(['features'], axis=1), axis = 1) 
    for idx in results.index:
        if results.loc[idx,'MIC_sum'] >= threshold_MIC:
            results.loc[idx,'MIC'] = True
        else:
            results.loc[idx,'MIC'] = False                          
    return results['MIC'].tolist()

def apply_selector(x_train, groups, y_train, features, selector=SelectFromModel(estimator = RandomForestRegressor(random_state = 42)), n_splits=5):
    results = pd.DataFrame({'features': features})
    kfold = fit_functions.custom_cv(x_train,y_train,groups,n_splits,1/n_splits)
    k = 0
    for train_idx, val_idx in kfold:
        k += 1
        x_train_new = np.array(x_train)[train_idx]
        y_train_new = np.array(y_train)[train_idx]
        scaler = MinMaxScaler()
        x_train_new = scaler.fit_transform(x_train_new)
        selector = selector
        selector.fit(x_train_new, y_train_new)
        results[f'SFM_{k}'] = selector.get_support()
    results['SFM_sum'] = np.sum(results.drop(['features'], axis=1), axis = 1)
    for idx in results.index:
        if results.loc[idx,'SFM_sum'] >= np.mean(results['SFM_sum']):
            results.loc[idx,'SFM'] = True
        else:
            results.loc[idx,'SFM'] = False
    return results['SFM'].tolist()

#%% Read data
df = pd.read_csv('../descriptors/mixture/main.csv')
df = add_descriptors.descriptors().add_several(df)
df = df.dropna(axis=1)
to_drop = ['Component#1','Smiles#1','Component#2','Smiles#2','T_EP','PD','Type']

y = df['T_EP']
x = df.drop(to_drop, axis = 1)
cv = fit_functions.custom_cv(x, y, df.groups, 1, 0.2)
for train_idx, val_idx in cv:
    x_train = x.iloc[train_idx]
    y_train = y.iloc[train_idx]
groups = x_train.groups
x_train_new = x_train.drop(['groups'],axis=1)

#%% Feature selection

selector = VarianceThreshold()
selector.fit_transform(x_train_new)
x_train_new = x_train_new.loc[:, selector.get_support()]

features = x_train_new.columns.tolist()                            
MIC_selection = pd.DataFrame({'features':features,'MIC':apply_MIC(x_train_new, groups, y_train, features, 0.2)})
selected = MIC_selection[MIC_selection['MIC']==True]['features'].tolist()
x_train_new = x_train_new[selected]

results = pd.DataFrame({'features':selected})
results['SFM_RF'] = apply_selector(x_train_new, groups, y_train, selected,selector=SelectFromModel(estimator = RandomForestRegressor(random_state = 42)))
results['MIC'] = apply_MIC(x_train_new, groups, y_train, selected, 0.35)
scaler = MinMaxScaler()
x_train_new = scaler.fit_transform(x_train_new)
selector = SequentialFeatureSelector(SVR(), direction='forward', n_features_to_select='auto', tol=0.1, scoring='neg_root_mean_squared_error', cv=fit_functions.custom_cv(x_train_new,y_train,groups,5,0.2), n_jobs=-1)
selector.fit(x_train_new, y_train)
results['SFS_SVR'] = selector.get_support()
results['sum'] = np.sum(results[['SFM_RF','MIC','SFS_SVR']], axis = 1)
for idx in results.index:
    if results.loc[idx,'sum'] >= 1:
        results.loc[idx,'final'] = 1
    else:
        results.loc[idx,'final'] = 0

results.to_csv('../results/feature_selection.csv', index=False)                                 
