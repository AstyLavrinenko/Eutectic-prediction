'''
Prediction of eutectic point
'''
#%% Imports
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

#%% Functions
def evaluate_model(y, y_pred):    
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared = False)
    accuracy = {'r2': r2, 'mae': mae, 'rmse': rmse}
    return accuracy

def fit_model(data_train, x_test, smiles_1, smiles_2, to_drop):
    
    # ML model for features_1
    model = RandomForestRegressor(n_estimators = 10, max_depth=4, min_samples_split=12, 
                                  min_samples_leaf=6, random_state = 42)
    # ML model for features_2
    model = RandomForestRegressor(n_estimators = 15, max_depth=8, min_samples_split=10, 
                                  min_samples_leaf=14, random_state = 42)
    
    y_train = data_train['T_EP']
    x_train = data_train.drop(to_drop, axis = 1)
    y_train = np.array(y_train).reshape(-1, 1)
        
    scaler_x = StandardScaler()
    x_train_new = scaler_x.fit_transform(np.array(x_train))
    x_test = scaler_x.transform(np.array(x_test))
    
    scaler_y = StandardScaler()
    y_train_new = np.ravel(scaler_y.fit_transform(y_train))
    
    model.fit(x_train_new, y_train_new)
    
    y_pred = model.predict(x_test)
    y_pred = scaler_y.inverse_transform(y_pred)
        
    return y_pred
#%% Read data
df = pd.DataFrame()
for csv in os.listdir('descriptors'):
    addend = pd.read_csv(f'descriptors/{csv}')
    df = pd.concat([df, addend], axis = 1)
df = df.loc[:,~df.columns.duplicated()]  
to_drop = ['Component_1', 'Smiles_1', 'Component_2', 'Smiles_2', 'T_EP']
features_1 = ['T_1', 'X_1', 'T_2']
features_2 = ['T_1', 'mu_1', 'H_MF_1', 'X_1', 'T_2', 'mu_2', 'H_int_2', 'mu_gas_2', 'MolLogP_2', 'WAPS_2', 'area_p_2']
features = features_1
model = 'model_1'
#features = features_2
#model = 'model_2'
#%% Extract the experimental data
dfs = [p.reset_index(drop=True) for _, p in df.groupby(by = ['Smiles_1', 'Smiles_2'])]
keys = []
values = []
for d_i in dfs:
    keys.append((d_i.loc[0, 'Smiles_1'], d_i.loc[0, 'Smiles_2']))
    value = pd.DataFrame(columns = ['T_EP', 'X_1'])
    value.loc[0, 'X_1'] = 0
    value.loc[1, 'X_1'] = 1
    value.loc[0, 'T_EP'] = d_i.loc[0, 'T_2']
    value.loc[1, 'T_EP'] = d_i.loc[0, 'T_1']
    value = pd.concat([value, d_i[['T_EP', 'X_1']]])
    values.append(value.sort_values(by=['X_1']))
experimental = dict(zip(keys, values))
#%% Algorithm for eutectic prediction
data = df[to_drop + features].sort_values(by = ['T_EP']).drop_duplicates(subset = ['Smiles_1', 'Smiles_2'], keep = 'first').sort_index()
results = data
concs = [i/20 for i in range(2,19)]
for idx in data.index:
    smiles_1 = data.loc[idx, 'Smiles_1']
    smiles_2 = data.loc[idx, 'Smiles_2']
    x_test = pd.DataFrame(columns = features)
    for num_conc, conc in enumerate(concs):
        for col in x_test.columns:
            if col == 'X_1':
                continue
            x_test.loc[num_conc, col] = data.loc[idx, col]
        x_test.loc[num_conc, 'X_1'] = conc
    x_test = x_test.convert_dtypes()
    y_pred = fit_model(df[to_drop + features].drop(index=idx), x_test, smiles_1, smiles_2, to_drop)
    # Plot results
    exp_data = experimental[(data.loc[idx, 'Smiles_1'], data.loc[idx, 'Smiles_2'])]
    plt.clf()
    plt.figure(figsize=(6,6))
    plt.plot(exp_data['X_1'], exp_data['T_EP'], alpha = 0.6, linestyle = '--', marker = 'o', markersize = 8, color='green', linewidth = 2, label='Experimental data')
    plt.scatter(x_test['X_1'], y_pred, s=70, facecolors='none', edgecolors='blue', linewidth = 2, label='Predicted by ML')
    plt.title(f'{data.loc[idx, "Component_1"]}_{data.loc[idx, "Component_2"]}', {'fontsize': 14})
    plt.xlabel(f'Mole fraction of {data.loc[idx,"Component_1"]}', {'fontsize': 14})
    plt.xticks(fontsize = 14)
    plt.ylabel('Temperature, K', {'fontsize': 14})
    plt.yticks(fontsize = 14)
    plt.legend(fontsize = 14, bbox_to_anchor=(1.04,0.9), loc='upper left', borderaxespad=0)
    plt.show()
    # Find the eutectic point
    y_min = y_pred[0]
    idx_min = 0
    for idx_y in range(len(y_pred)):
        if y_pred[idx_y] < y_min:
            y_min = y_pred[idx_y]
            idx_min = idx_y
    results.loc[idx, 'T_ML'] = y_min
    results.loc[idx, 'X_ML'] = x_test.loc[idx_min, 'X_1']
# Calculate accuracy of prediction
accuracy_T = evaluate_model(np.array(results['T_EP']), np.array(results['T_ML']))    
accuracy_X = evaluate_model(np.array(results['X_1']), np.array(results['X_ML']))

results.to_csv(f'results_{model}.csv', index = False)