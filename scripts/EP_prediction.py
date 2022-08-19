'''
Prediction of eutectic point
'''
#%% Imports
import os
import pandas as pd
import numpy as np

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
df = pd.DataFrame()
#for csv in os.listdir('../descriptors'):
for csv in ['a_thermochem.csv', 'cheminfo_2D.csv']:
    addend = pd.read_csv(f'../descriptors/{csv}')
    df = pd.concat([df, addend], axis = 1)
df = df.loc[:,~df.columns.duplicated()]
df=df.dropna(axis=1)
#df_2D=df
#df = pd.read_csv('../descriptors/a_thermochem.csv')
#df=df.dropna(axis=1)
df['ln_x#1'] = np.log(df['X#1'])
df['ln_x#2'] = np.log(1-df['X#1'])
df['frac'] = df['X#1']/(1-df['X#1'])
df['T_frac'] = df['X#1']*df['T#1']+(1-df['X#1'])*df['T#2']
df['T_frac#1'] = df['X#1']*df['T#1']
df['T_frac#2'] = (1-df['X#1'])*df['T#2']
to_drop = ['Component#1', 'Smiles#1', 'Component#2', 'Smiles#2', 'T_EP', 'Phase_diagram', 'inchi#1', 'inchi#2','X#1_init']
#cols_2D=[]
#for col in df_2D.columns:
#    if col in to_drop+['X#1','Type_I','Type_III','Type_V','Solubility#1','Solubility#2']:
#        continue
#    col = col.split('#')[0]
#    if col in cols_2D:
#        continue
#    cols_2D.append(col)
#    df[col] = (df_2D['X#1'])*df_2D[f'{col}#1']+(1-df_2D['X#1'])*df_2D[f'{col}#2']
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
    values.append(value.sort_values(by=['X#1']))
experimental = dict(zip(keys, values))
#%% Algorithm for eutectic prediction

data = df.loc[df['Phase_diagram'] == 'Yes'].sort_values(by = ['T_EP']).drop_duplicates(subset = ['Smiles#1', 'Smiles#2'], keep = 'first').sort_index()
results = data[['Component#1', 'Smiles#1', 'Component#2', 'Smiles#2', 'T_EP', 'X#1']]
concs = [i/20 for i in range(2,19)]
for idx in data.index:
    smiles_1 = data.loc[idx, 'Smiles#1']
    smiles_2 = data.loc[idx, 'Smiles#2']
    x_test = pd.DataFrame(columns = features)
    for num_conc, conc in enumerate(concs):
        for col in x_test.columns:
            if col in ['X#1', 'Solubility#1','Solubility#2','ln_x#1','ln_x#2','frac','bins','T_frac','T_frac#1','T_frac#2']:
                continue
            x_test.loc[num_conc, col] = data.loc[idx, col]
        x_test.loc[num_conc, 'X#1'] = conc
        x_test.loc[num_conc, 'ln_x#1'] = np.log(conc)
        x_test.loc[num_conc, 'ln_x#2'] = np.log(1-conc)
        MolWt1=Descriptors.MolWt(Chem.MolFromSmiles(smiles_1))
        MolWt2=Descriptors.MolWt(Chem.MolFromSmiles(smiles_2))
        x_test.loc[num_conc, 'Solubility#1'] = 100*MolWt1*conc/(MolWt2*(1-conc))
        x_test.loc[num_conc, 'Solubility#2'] = 100*MolWt2*(1-conc)/(MolWt2*conc)
        x_test.loc[num_conc, 'frac'] = conc/(1-conc)
        x_test.loc[num_conc, 'T_frac'] = conc*x_test.loc[num_conc, 'T#1']+(1-conc)*x_test.loc[num_conc, 'T#2']
        x_test.loc[num_conc, 'T_frac#1'] = conc*x_test.loc[num_conc, 'T#1']
        x_test.loc[num_conc, 'T_frac#2'] = (1-conc)*x_test.loc[num_conc, 'T#2']
        #for col in cols_2D:            
        #    x_test.loc[num_conc, col] = conc*df_2D.loc[idx, f'{col}#1']+(1-conc)*df_2D.loc[idx, f'{col}#2']
    x_test['bins'] = np.digitize(x_test['X#1'], bins=[0,0.05,0.2,0.35,0.5,0.65,0.8,0.95,1])
    x_test = x_test.convert_dtypes()
    y_pred = fit_model(df.drop(index=df.loc[(df['Smiles#1'] == smiles_1) & (df['Smiles#2'] == smiles_2)].index), x_test, smiles_1, smiles_2, to_drop)
    def f1(x,a,b,c):
        f=a*abs(1-x-x)+b*np.exp(x/(1-x))+c
        return f
    def f2(x,a,b,c):
        return a*x**2+b*x+c
    popt, _ = curve_fit(f1, np.array(x_test['X#1']), np.array(y_pred))
    a,b,c = popt
    def f1_opt(x):
        f=a*abs(1-x-x)+b*np.exp(x/(1-x))+c
        return f
    y1 = f1(x_test['X#1'],a,b,c)
    x1_min = minimize_scalar(f1_opt,bounds=(0.1, 0.8),method='bounded')['x']
    x1 = [abs(conc-x1_min) for conc in concs]
    results.loc[idx, 'X_ML1'] = concs[x1.index(np.min(x1))]
    results.loc[idx, 'T_ML1'] = y_pred[x1.index(np.min(x1))]
    popt, _ = curve_fit(f2, np.array(x_test['X#1']), np.array(y_pred))
    a,b,c = popt
    def f2_opt(x):
        return a*x**2+b*x+c
    y2 = f2(x_test['X#1'],a,b,c)
    x2_min = minimize_scalar(f1_opt,bounds=(0.1, 0.8),method='bounded')['x']
    x2 = [abs(conc-x2_min) for conc in concs]
    results.loc[idx, 'X_ML2'] = concs[x2.index(np.min(x2))]
    results.loc[idx, 'T_ML2'] = y_pred[x2.index(np.min(x2))]
    # Plot results
    exp_data = experimental[(data.loc[idx, 'Smiles#1'], data.loc[idx, 'Smiles#2'])]
    plt.clf()
    plt.figure(figsize=(6,6))
    plt.plot(exp_data['X#1'], exp_data['T_EP'], alpha = 0.6, linestyle = '--', marker = 'o', markersize = 8, color='green', linewidth = 2, label='Experimental data')
    plt.scatter(x_test['X#1'], y_pred, s=70, facecolors='none', edgecolors='blue', linewidth = 2, label='Predicted by ML')
    plt.scatter(concs[x1.index(np.min(x1))], y_pred[x1.index(np.min(x1))], s=70, facecolors='none', edgecolor='orange', linewidth = 2, label='f1')
    plt.scatter(concs[x2.index(np.min(x2))], y_pred[x2.index(np.min(x2))], s=70, facecolors='none', edgecolor='purple', linewidth = 2, label='f2')
    plt.title(f'{data.loc[idx, "Component#1"]}_{data.loc[idx, "Component#2"]}', {'fontsize': 14})
    plt.xlabel(f'Mole fraction of {data.loc[idx,"Component#1"]}', {'fontsize': 14})
    plt.xticks(fontsize = 14)
    plt.ylabel('Temperature, K', {'fontsize': 14})
    plt.yticks(fontsize = 14)
    # Find the eutectic point
    y_min = y_pred[0]
    idx_min = 0
    for idx_y in range(len(y_pred)):
        if x_test.loc[idx_y, 'X#1'] in [0.05,0.1,0.9,0.95]:
            continue
        if y_pred[idx_y] < y_min:
            y_min = y_pred[idx_y]
            idx_min = idx_y
    results.loc[idx, 'T_ML'] = y_min
    results.loc[idx, 'X_ML'] = x_test.loc[idx_min, 'X#1']
    plt.scatter(x_test.loc[idx_min, 'X#1'], y_min, s=70, facecolors='none', edgecolors='red', linewidth = 2, label='Predicted EP')
    plt.legend(fontsize = 14, bbox_to_anchor=(1.04,0.9), loc='upper left', borderaxespad=0)
    plt.show()
# Calculate accuracy of prediction
accuracy_T = evaluate_model(np.array(results['T_EP']), np.array(results['T_ML']))    
accuracy_X = evaluate_model(np.array(results['X#1']), np.array(results['X_ML']))
accuracy_T1 = evaluate_model(np.array(results['T_EP']), np.array(results['T_ML1']))    
accuracy_X1 = evaluate_model(np.array(results['X#1']), np.array(results['X_ML1']))
accuracy_T2 = evaluate_model(np.array(results['T_EP']), np.array(results['T_ML2']))    
accuracy_X2 = evaluate_model(np.array(results['X#1']), np.array(results['X_ML2']))
