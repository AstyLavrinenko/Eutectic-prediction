''' feature selection '''
#%% Imports
import fit_evaluate
import add_descriptors

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from minepy import MINE
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold

#%% Read data
df=pd.read_csv('../descriptors/mixture/main.csv')

df=add_descriptors.descriptors().add_several(df, profile=False, potential=False)
df=add_descriptors.descriptors().add_profile(df, conditions={add_descriptors.ratio:'all'}, flag=False)
df=add_descriptors.descriptors().add_potent(df, conditions={add_descriptors.ratio:'all'}, flag=False)
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

df = df.drop(['MolWeight#1','MolWeight#2','Area#1','Area#2','Volume#1','Volume#2',
              'NumValenceElectrons#1','NumValenceElectrons#2'], axis=1)

df = df.dropna(axis=1)
to_drop = ['Component#1', 'Smiles#1', 'Component#2', 'Smiles#2', 'T_EP', 'Phase_diagram']

train_idx, test_idx = fit_evaluate.split_by_bins(df, 0.2)
y = 1/df['T_EP']
x = df.drop(to_drop, axis = 1)
x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
features_save= ['X#1', 'T#1', 'T#2'] + [f'cluster_{i}' for i in range(20)]
#%% Function
def fit_selector(f_selector, x_train, y_train, features_save, threshold, n_splits):
    features = x_train.columns.tolist()
    results = pd.DataFrame({'features': features})
    kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 42)
    k = 0
    for train_idx, val_idx in kfold.split(x_train):
        k += 1
        x_train_new, x_val = np.array(x_train)[train_idx], np.array(x_train)[val_idx]
        y_train_new, y_val = np.array(y_train)[train_idx], np.array(y_train)[val_idx]
        
        scaler = MinMaxScaler()
        x_train_new = scaler.fit_transform(x_train_new)
        x_val = scaler.transform(x_val)
        
        if f_selector == 'MIC':
            df_mic = pd.DataFrame({'features': features})
            for idx in results.index:
                mine = MINE(alpha=0.6, c=15, est="mic_approx")
                mine.compute_score(x_train_new[:,idx], y_train_new)
                df_mic.loc[idx, 'MIC'] = mine.mic()
            mean = np.mean(df_mic['MIC'])
            print('mean ', mean)
            for idx in results.index:
                feature = results.loc[idx, 'features']
                if feature in features_save:
                    results.loc[idx, f'fold_{k}'] = True
                else:
                    if df_mic.loc[idx, 'MIC'] >= threshold:
                        results.loc[idx, f'fold_{k}'] = True
                    else:
                        results.loc[idx, f'fold_{k}'] = False       
        else:
            selector = f_selector
            selector.fit(x_train_new, y_train_new)
            results[f'fold_{k}'] = selector.get_support()
            print(k)
    results['sum'] = np.sum(results.drop(['features'], axis=1), axis = 1)
    threshold_2 = np.mean(results['sum'])
    for idx in results.index:
        if results.loc[idx, 'sum'] >= threshold_2:
            results.loc[idx, 'final'] = 1
        else:
            results.loc[idx, 'final'] = 0
    return(results)

#%% feature selection
selector = VarianceThreshold()
selector.fit_transform(x_train)
selected = x_train.loc[:, selector.get_support()].columns.tolist()

#for threshold in [0.35]:
#    print(len(selected))
#    x_train_new = x_train[selected]
#    params_mic = {'f_selector':'MIC', 'x_train':x_train_new, 'y_train':y_train, 'features_save':[], 'threshold':threshold}
#    results = fit_selector(**params_mic)
#    selected = results[results['final'] == 1]['features'].tolist()
#    print(len(selected))
#    results.to_csv(f'../results/MIC_{threshold}.csv', index=False)
#x_train[selected].corr().to_csv('../results/corr.csv')


params_mic = {'f_selector':'MIC', 'x_train':x_train, 'y_train':y_train, 'features_save':[], 'threshold':0.3, 'n_splits':6}
params_sfs = {'f_selector': SequentialFeatureSelector(GradientBoostingRegressor(random_state=42), n_features_to_select='auto', tol=0.05, scoring='r2', cv=2, n_jobs=-1), 'x_train':x_train, 'y_train':y_train, 'features_save':[], 'threshold':0.35, 'n_splits':3}
selectors = {'MIC': params_mic, 'sfs': params_sfs}
selected = pd.DataFrame({'features': x_train.columns.tolist()})
features_new = []
for name, params in selectors.items():
    results = fit_selector(**params)
    results.to_csv(f'../results/{name}.csv', index=False)
    selected[f'{name}'] = results['final']
selected['sum'] = selected.sum(axis = 1, numeric_only=True)
selected = selected.sort_values(by = ['sum'], ascending=False)
threshold = np.mean(selected['sum'])
for idx in selected.index:
    if selected.loc[idx, 'sum'] >= threshold:
        features_new.append(selected.loc[idx, 'features'])
x_train[features_new].corr().to_csv('../results/corr.csv')

