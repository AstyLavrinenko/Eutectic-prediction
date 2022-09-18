#%% Imports
import fit_evaluate
import add_descriptors

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from minepy import MINE
from sklearn.feature_selection import VarianceThreshold

#%% Functions
def fit_selector(f_selector, x_train, y_train, features_save, threshold):
    features = x_train.columns.tolist()
    results = pd.DataFrame({'features': features})
    kfold = KFold(n_splits = 10, shuffle = True, random_state = 42)
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
            for idx in results.index:
                feature = results.loc[idx, 'features']
                if feature in features_save:
                    results.loc[idx, f'fold_{k}'] = True
                else:
                    if (df_mic.loc[idx, 'MIC'] >= mean-threshold) & (df_mic.loc[idx, 'MIC'] <= mean+threshold):
                        results.loc[idx, f'fold_{k}'] = True
                    else:
                        results.loc[idx, f'fold_{k}'] = False       
        else:
            selector = f_selector
            selector.fit(x_train_new, y_train_new)
            results[f'fold_{k}'] = selector.get_support()
    results['sum'] = np.sum(results.drop(['features'], axis=1), axis = 1)
    threshold_2 = np.mean(results['sum'])
    for idx in results.index:
        if results.loc[idx, 'sum'] >= threshold_2:
            results.loc[idx, 'final'] = 1
        else:
            results.loc[idx, 'final'] = 0
    return(results)

def select_function():
    results = pd.DataFrame(columns=['features', 'without', 'ratio', 'log_ratio', 'deg'])
    results_dict = {}
    for idx in df.index:
        feature = df.loc[idx, 'features'].split('#')[0]
        func = df.loc[idx, 'features'].split('#')[1]
        results_dict[feature] = []
        

#%% Main

df=pd.read_csv('../descriptors/mixture/main.csv')

df_2D=add_descriptors.descriptors().add_2D(df, None, [])
df_2D=add_descriptors.descriptors().add_2D(df_2D, add_descriptors.ratio, 'all')
df_2D=add_descriptors.descriptors().add_2D(df_2D, add_descriptors.deg, 'all')
df_2D=add_descriptors.descriptors().add_2D(df_2D, add_descriptors.log_ratio, 'all')

df_mom=add_descriptors.descriptors().add_mom(df, None, [])
df_mom=add_descriptors.descriptors().add_mom(df_mom, add_descriptors.ratio, 'all')
df_mom=add_descriptors.descriptors().add_mom(df_mom, add_descriptors.deg, 'all')
df_mom=add_descriptors.descriptors().add_mom(df_mom, add_descriptors.log_ratio, 'all')

df_mu=add_descriptors.descriptors().add_mu_05(df, None, [])
df_mu=add_descriptors.descriptors().add_mu_05(df_mu, add_descriptors.deg, 'all')
df_mu=add_descriptors.descriptors().add_mu_05(df_mu, add_descriptors.log_ratio, 'all')
df_mu=add_descriptors.descriptors().add_mu_inf(df_mu, None, [])
df_mu=add_descriptors.descriptors().add_mu_inf(df_mu, add_descriptors.deg, 'all')
df_mu=add_descriptors.descriptors().add_mu_inf(df_mu, add_descriptors.log_ratio, 'all')

for key, value in {'2D':df_2D, 'mom':df_mom, 'mu':df_mu}.items():
    value = value.dropna(axis=1)
    to_drop = ['Component#1', 'Smiles#1', 'Component#2', 'Smiles#2', 'T_EP', 'Phase_diagram']

    train_idx, test_idx = fit_evaluate.split_by_bins(value, 0.2)
    y = value['T_EP']
    x = value.drop(to_drop, axis = 1)
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    selector = VarianceThreshold()
    selector.fit_transform(x_train)
    selected = x_train.loc[:, selector.get_support()].columns.tolist()
    for threshold in [0.1, 0.45]:
        print(len(selected))
        x_train_new = x_train[selected]
        params_mic = {'f_selector':'MIC', 'x_train':x_train_new, 'y_train':y_train, 'features_save':[], 'threshold':threshold}
        results = fit_selector(**params_mic)
        selected = results[results['final'] == 1]['features'].tolist()
        results.to_csv(f'../results/MIC_{key}_{threshold}.csv', index=False)
        print(len(selected))
    x_train[selected].corr().to_csv(f'../results/corr_{key}.csv')