'''
Feature selection
'''
#%% Imports
import fit_evaluate
import add_descriptors

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from minepy import MINE
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, SequentialFeatureSelector

#%% Read data
df=pd.read_csv('../descriptors/mixture/main.csv')

df=add_descriptors.descriptors().add_several(df)
df=add_descriptors.descriptors().add_several(df, conditions={add_descriptors.log_ratio:'all'})
df=add_descriptors.descriptors().add_several(df, conditions={add_descriptors.deg:'all'})
df['ln_x1'] = np.log(df['X#1'])
df['ln_x2'] = np.log(1-df['X#1'])
df['MW_per_Vol#1'] = df['MolWeight#1']/df['Volume#1']
df['MW_per_Vol#2'] = df['MolWeight#2']/df['Volume#2']
df['ValE_per_Area#1'] = df['NumValenceElectrons#1']/df['Area#1']
df['ValE_per_Area#2'] = df['NumValenceElectrons#2']/df['Area#2']
pi = 3.14159265358979323846264338328
df['R#1'] = ((df['Area#1']/(4*pi))**0.5 + (3*df['Volume#1']/(4*pi))**(1/3))/2
df['R#2'] = ((df['Area#2']/(4*pi))**0.5 + (3*df['Volume#2']/(4*pi))**(1/3))/2
df['HBD-HBA#1'] = df['NumHDonors#1']-df['NumHAcceptors#1']
df['HBD-HBA#2'] = df['NumHDonors#2']-df['NumHAcceptors#2']
df['HBD+HBA#1'] = df['NumHDonors#1']+df['NumHAcceptors#1']
df['HBD+HBA#2'] = df['NumHDonors#2']+df['NumHAcceptors#2']
df['1/T#1'] = 1/df['T#1']
df['1/T#2'] = 1/df['T#2']

df = df.dropna(axis=1)
to_drop = ['Component#1', 'Smiles#1', 'Component#2', 'Smiles#2', 'T_EP', 'Phase_diagram']

train_idx, test_idx = fit_evaluate.split_by_bins(df, 0.2)
y = df['T_EP']
x = df.drop(to_drop, axis = 1)
x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
#features_save= ['X#1', 'T#1', 'T#2'] + [f'cluster_{i}' for i in range(20)]

#%% Feature selection
print(len(x_train.columns))
selector = VarianceThreshold()
selector.fit_transform(x_train)
x_train = x_train.loc[:, selector.get_support()]
print(len(x_train.columns))
scaler = MinMaxScaler()
x_train_new = scaler.fit_transform(x_train)
selector = SequentialFeatureSelector(GradientBoostingRegressor(random_state=42), direction='forward', n_features_to_select=0.25, scoring='neg_root_mean_squared_error', cv=3, n_jobs=-1)
selector.fit(x_train_new, y_train)
x_train = x_train.loc[:, selector.get_support()]
print(len(x_train.columns))
features = x_train.columns.tolist()
results = pd.DataFrame({'features': features})
kfold = KFold(n_splits = 5, shuffle = True, random_state = 42)
k = 0
threshold_MIC = 0.35
for train_idx, val_idx in kfold.split(x_train):
    k += 1
    x_train_new, x_val = np.array(x_train)[train_idx], np.array(x_train)[val_idx]
    y_train_new, y_val = np.array(y_train)[train_idx], np.array(y_train)[val_idx]
    scaler = MinMaxScaler()
    x_train_new = scaler.fit_transform(x_train_new)
    for idx in results.index:
        selector = MINE(alpha=0.6, c=15, est="mic_approx")
        selector.compute_score(x_train_new[:,idx], y_train_new)
        if selector.mic() >= threshold_MIC:
            results.loc[idx, f'MIC_{k}'] = True
        else:
            results.loc[idx, f'MIC_{k}'] = False
    selector = SelectFromModel(estimator = GradientBoostingRegressor(random_state = 42))
    selector.fit(x_train_new, y_train_new)
    results[f'SFM_{k}'] = selector.get_support()
    print(k)
results['sum'] = np.sum(results.drop(['features'], axis=1), axis = 1)
threshold_mean = np.mean(results['sum'])
results[results['sum'] >= threshold_mean]['final'] = 1
results[results['sum'] < threshold_mean]['final'] = 0
results.to_csv('../results/feature_selection_step2.csv')                                 
selected = results[results['final'] == 1]['features'].tolist()
x_train[selected].corr().to_csv('../results/corr.csv')