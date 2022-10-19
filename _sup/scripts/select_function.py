''' Select function '''

import fit_evaluate
import add_descriptors

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from minepy import MINE

#%% Read data
df=pd.read_csv('../descriptors/mixture/main.csv')
'''
2D - ratio
InfD - without
mom - sqr_ratio
potent - sqr_ratio
profile - ratio
'''
df=add_descriptors.descriptors().add_several(df)
df=add_descriptors.descriptors().add_several(df, conditions={add_descriptors.log_ratio:'all'}, thermo=False, cluster=False)
df=add_descriptors.descriptors().add_several(df, conditions={add_descriptors.ratio:'all'}, thermo=False, cluster=False)
df=add_descriptors.descriptors().add_several(df, conditions={add_descriptors.squared_ratio:'all'}, thermo=False, cluster=False)
df=add_descriptors.descriptors().add_several(df, conditions={add_descriptors.deg:'all'}, thermo=False, cluster=False)
df=add_descriptors.descriptors().add_several(df, conditions={add_descriptors.difference:'all'}, thermo=False, cluster=False)
df['ln_gamma_x1'] = np.log(df['X#1'])+df['ln_gamma_InfD#1']
df['ln_gamma_x2'] = np.log(1-df['X#1'])+df['ln_gamma_InfD#2']
df['MW_per_Vol#1'] = df['MolWeight#1']/df['Volume#1']
df['MW_per_Vol#2'] = df['MolWeight#2']/df['Volume#2']
df['gamma'] = (np.exp(df['ln_gamma_InfD#1'])**df['X#1'])*(np.exp(df['ln_gamma_InfD#2'])**(1-df['X#1']))
df['ValE_per_Area#1'] = df['NumValenceElectrons#1']/df['Area#1']
df['ValE_per_Area#2'] = df['NumValenceElectrons#2']/df['Area#2']
pi = 3.14159265358979323846264338328
df['R#1'] = ((df['Area#1']/(4*pi))**0.5 + (3*df['Volume#1']/(4*pi))**(1/3))/2
df['R#2'] = ((df['Area#2']/(4*pi))**0.5 + (3*df['Volume#2']/(4*pi))**(1/3))/2
df['HBD-HBA'] = df['X#1']*(df['NumHDonors#1']-df['NumHAcceptors#1'])+(1-df['X#1'])*(df['NumHAcceptors#2']-df['NumHDonors#2'])

df = df.dropna(axis=1)
to_drop = ['Component#1','Smiles#1','Component#2','Smiles#2','T_EP','Phase_diagram','bins']

train_idx, test_idx = fit_evaluate.split_by_bins(df, 0.2)
y = df['T_EP']
x = df.drop(to_drop, axis = 1)
x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

#%%
features = x_train.columns.tolist()
results = pd.DataFrame({'features': features})
kfold = KFold(n_splits = 5, shuffle = True, random_state = 42)
k = 0
for train_idx, val_idx in kfold.split(x_train):
    k += 1
    x_train_new, x_val = np.array(x_train)[train_idx], np.array(x_train)[val_idx]
    y_train_new, y_val = np.array(y_train)[train_idx], np.array(y_train)[val_idx]
    scaler = MinMaxScaler()
    x_train_new = scaler.fit_transform(x_train_new)
    for idx in results.index:
        selector = MINE(alpha=0.6, c=15, est="mic_approx")
        selector.compute_score(x_train_new[:,idx], y_train_new)
        results.loc[idx, f'MIC_{k}'] = selector.mic()
results.to_csv('../results/MIC.csv')