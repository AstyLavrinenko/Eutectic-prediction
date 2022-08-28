'''
Prediction of eutectic point
'''
#%% Imports
import os
import pandas as pd
import numpy as np
import add_descriptors

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit, minimize_scalar

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors

#%% Functions
def evaluate_model(y, y_pred):    
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared = False)
    accuracy = {'r2': r2, 'mae': mae, 'rmse': rmse}
    return accuracy

def fit_model(data_train, x_test, smiles_1, smiles_2, to_drop):

    model = GradientBoostingRegressor()
    
    y_train = data_train['T_EP']
    x_train = data_train.drop(to_drop, axis = 1)
    y_train = np.array(y_train)
        
    scaler_x = MinMaxScaler()
    x_train_new = scaler_x.fit_transform(np.array(x_train))
    x_test = scaler_x.transform(np.array(x_test))
    
    model.fit(x_train_new, y_train)
    y_pred = model.predict(x_test)
        
    return y_pred    
#%% Read data
df=pd.read_csv('../descriptors/mixture/main.csv')
df=add_descriptors.descriptors().add_several(df)
df['ln_x#1'] = np.log(df['X#1'])
df['ln_x#2'] = np.log(1-df['X#1'])
df['frac'] = df['X#1']/(1-df['X#1'])
df['T_frac'] = df['X#1']*df['T#1']+(1-df['X#1'])*df['T#2']
df['T_frac#1'] = df['X#1']*df['T#1']
df['T_frac#2'] = (1-df['X#1'])*df['T#2']
to_drop = ['Component#1','Smiles#1','Component#2','Smiles#2','T_EP','H#1','H#2','Kf#1','Kf#2','Phase_diagram']
features = list(df.drop(to_drop, axis=1).columns)
#%% Extract the experimental data
dfs = [p.reset_index(drop=True) for _, p in df.groupby(by = ['Smiles#1', 'Smiles#2'])]
keys = []
values = []
for d_i in dfs:
    keys.append((d_i.loc[0, 'Smiles#1'], d_i.loc[0, 'Smiles#2']))
    value = pd.DataFrame(columns = ['T_EP', 'X#1'])
    value.loc[0, 'X#1'] = 0
    value.loc[1, 'X#1'] = 1
    value.loc[0, 'T_EP'] = d_i.loc[0, 'T#2']
    value.loc[1, 'T_EP'] = d_i.loc[0, 'T#1']
    value = pd.concat([value, d_i[['T_EP', 'X#1']]])
    value['X#1'] = pd.to_numeric(value['X#1'])
    value['T_EP'] = pd.to_numeric(value['T_EP'])
    values.append(value.sort_values(by=['X#1']).reset_index(drop=True))
experimental = dict(zip(keys, values))
#%% Algorithm for eutectic prediction
data = df[df['Phase_diagram']=='Yes'].sort_values(by = ['T_EP']).drop_duplicates(subset = ['Smiles#1', 'Smiles#2'], keep = 'first').sort_index()
results = data[['Component#1', 'Smiles#1', 'Component#2', 'Smiles#2', 'T_EP', 'X#1']]
concs = [i/20 for i in range(2,19)]
for idx in data.index:
    smiles_1 = data.loc[idx, 'Smiles#1']
    smiles_2 = data.loc[idx, 'Smiles#2']
    x_test = pd.DataFrame(columns = to_drop+features)
    x_test['X#1'] = concs
    x_test['Smiles#1'] = smiles_1
    x_test['Smiles#2'] = smiles_2
    x_test = add_descriptors.descriptors().add_several(x_test)
    MolWt1=Descriptors.MolWt(Chem.MolFromSmiles(smiles_1))
    MolWt2=Descriptors.MolWt(Chem.MolFromSmiles(smiles_2))
    x_test['Solubility#1'] = 100*MolWt1*x_test['X#1']/(MolWt2*(1-x_test['X#1']))
    x_test['Solubility#2'] = 100*MolWt2*(1-x_test['X#1'])/(MolWt1*x_test['X#1'])
    x_test['ln_x#1'] = np.log(x_test['X#1'])
    x_test['ln_x#2'] = np.log(1-x_test['X#1'])
    x_test['frac'] = x_test['X#1']/(1-x_test['X#1'])
    x_test['T_frac'] = x_test['X#1']*x_test['T#1']+(1-x_test['X#1'])*x_test['T#2']
    x_test['T_frac#1'] = x_test['X#1']*x_test['T#1']
    x_test['T_frac#2'] = (1-x_test['X#1'])*x_test['T#2']
    x_test['bins'] = np.digitize(x_test['X#1'], bins=[0]+[round(i,2) for i in np.linspace(0.15,0.85,8)]+[1])
    x_test = x_test.drop(to_drop, axis=1)
    y_pred = fit_model(df.drop(index=df[(df['Smiles#1'] == smiles_1) & (df['Smiles#2'] == smiles_2)].index), x_test, smiles_1, smiles_2, to_drop)
    x_test['y_pred'] = y_pred
    y_min = np.min(x_test[(x_test['X#1'] != 0.95) & (x_test['X#1'] != 0.9) & (x_test['X#1'] != 0.05) & (x_test['X#1'] != 0.1)]['y_pred'])
    x_min = x_test.loc[x_test[x_test['y_pred'] == y_min].index[0], 'X#1']
    results.loc[idx, 'T_ML'] = y_min
    results.loc[idx, 'X_ML'] = x_min
    # Plot results
    exp_data = experimental[(data.loc[idx, 'Smiles#1'], data.loc[idx, 'Smiles#2'])]
    plt.clf()
    plt.figure(figsize=(6,6))
    plt.plot(exp_data['X#1'], exp_data['T_EP'], alpha = 0.6, linestyle = '--', marker = 'o', markersize = 8, color='green', linewidth = 2, label='Experimental data')
    plt.scatter(x_test['X#1'], y_pred, s=70, facecolors='none', edgecolors='blue', linewidth = 2, label='Predicted by ML')
    plt.scatter(data.loc[idx,'X#1'], data.loc[idx,'T_EP'], s=70, facecolors='black', linewidth = 2, label='EP')
    plt.title(f'{data.loc[idx, "Component#1"]}_{data.loc[idx, "Component#2"]}', {'fontsize': 14})
    plt.xlabel(f'Mole fraction of {data.loc[idx,"Component#1"]}', {'fontsize': 14})
    plt.xticks(fontsize = 14)
    plt.ylabel('Temperature, K', {'fontsize': 14})
    plt.yticks(fontsize = 14)
    #plt.ylim([200, 620])
    # Find the eutectic point
    plt.scatter(x_min, y_min, s=70, facecolors='none', edgecolors='red', linewidth = 2, label='Predicted EP')
    plt.legend(fontsize = 14, bbox_to_anchor=(1.04,0.9), loc='upper left', borderaxespad=0)
    plt.show()
# Calculate accuracy of prediction
accuracy_T = evaluate_model(np.array(results['T_EP']), np.array(results['T_ML']))    
accuracy_X = evaluate_model(np.array(results['X#1']), np.array(results['X_ML']))
