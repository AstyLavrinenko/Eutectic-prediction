
import add_descriptors

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

#%% Functions
def evaluate_model(y, y_pred):    
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared = False)
    mape = mean_absolute_percentage_error(y, y_pred)
    accuracy = {'r2': r2, 'mae': mae, 'rmse': rmse, 'mape':mape}
    return accuracy

def fit_model(x_train, y_train, x_test, model_name):
    y_train_new = np.array(y_train).reshape(-1, 1)
    scaler = MinMaxScaler()
    x_train_new = scaler.fit_transform(np.array(x_train))
    x_test = scaler.transform(np.array(x_test))
    
    if model_name == 'GBR':
        model = GradientBoostingRegressor(learning_rate=0.2, max_depth=6,
                        max_features=8, min_samples_leaf=0.04,
                        min_samples_split=0.1, n_estimators=45)
    if model_name == 'RFR':
        model = RandomForestRegressor(max_depth=9, max_features=5, min_samples_leaf=0.02,
                              min_samples_split=0.05, n_estimators=65, n_jobs=-1)
    if model_name == 'KNN':
        model = KNeighborsRegressor(metric='manhattan', n_neighbors=7)
    if model_name == 'SVR':
        scaler_y = StandardScaler()        
        y_train_new = np.ravel(scaler_y.fit_transform(y_train_new))
        model = SVR(C=5.0, gamma=1.0)
    model.fit(x_train_new,np.ravel(y_train_new))
    y_pred = model.predict(x_test)
    if model_name == 'SVR':
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(1,-1)[0]
    return y_pred

def find_minT(x,y):
    y_min = np.min(y[2:-2])
    x_min = x[y.index(y_min)]
    return x_min, y_min

def plot_sle(x_test,pred_data,x_ep,t_ep,component_1,component_2,exp_data,colors={'GBR':'blueviolet',
                                                                                     'RFR':'gold',
                                                                                     'SVR':'maroon',
                                                                                     'KNN':'salmon'}):
    plt.clf()
    plt.plot(exp_data['X#1'], exp_data['T_EP'], alpha = 0.6, linestyle = '--', marker = 'o', markersize = 8, color='green', linewidth = 2, label='Experimental data')
    for name,data in pred_data.items():
        plt.scatter(x_test['X#1'], data['y_pred'], s=70, facecolors='none', edgecolors=colors[name], linewidth = 2, label=f'{name}')
        plt.scatter(data['x_min_1'], data['y_min_1'], s=70, facecolors=colors[name], linewidth = 2)
    plt.scatter(x_ep, t_ep, s=70, facecolors='black', linewidth = 2)
    plt.title(f'{component_1}_{component_2}', {'fontsize': 14})
    plt.xlabel(f'Mole fraction of {component_1}', {'fontsize': 14})
    plt.xticks(fontsize = 14)
    plt.ylabel('Temperature, K', {'fontsize': 14})
    plt.yticks(fontsize = 14)
    plt.legend(fontsize = 14, bbox_to_anchor=(1.04,0.9), loc='upper left', borderaxespad=0)
    plt.show()
    return

#%% Read data and add descriptors
df = pd.read_csv('../descriptors/mixture/main.csv')
results = df.copy().sort_values(by = ['T_EP']).drop_duplicates(subset = ['Smiles#1', 'Smiles#2'], keep = 'first').sort_index()
df = add_descriptors.descriptors().add_selected_features(df)
to_drop = ['Component#1','Smiles#1','Component#2','Smiles#2','T_EP','PD','groups']
df = pd.get_dummies(df, columns = ['Type'])

#%% Get experimental SLE data
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

#%% Prediction of eutectic point
concs = [i/20 for i in range(2,19)]
for idx in results.index:
    smiles_1 = results.loc[idx, 'Smiles#1']
    smiles_2 = results.loc[idx, 'Smiles#2']
    x_test = pd.DataFrame(columns = to_drop)
    x_test['X#1'] = concs
    x_test['Smiles#1'] = smiles_1
    x_test['Smiles#2'] = smiles_2
    x_type = df.loc[idx,['Type_IL mixture','Type_Type III','Type_Type V']]
    x_test = add_descriptors.descriptors().add_selected_features(x_test)
    x_test[['Type_IL mixture','Type_Type III','Type_Type V']] = x_type
    x_train = df.drop(index=df[(df['groups']==results.loc[idx,'groups'])].index)
    y_train = df.drop(index=df[(df['groups']==results.loc[idx,'groups'])].index)['T_EP']
    predicted_data = {}
    for model in ['GBR','RFR','SVR','KNN']:
        y_pred = fit_model(x_train.drop(to_drop,axis=1),y_train,x_test.drop(to_drop,axis=1),model)
        x_min_1, y_min_1 = find_minT(x_test['X#1'].tolist(), y_pred.tolist())
        results.loc[idx, f'T_{model}'] = y_min_1
        results.loc[idx, f'X_{model}'] = x_min_1
        predicted_data[model] = {'y_pred':y_pred,'y_min_1':y_min_1,'x_min_1':x_min_1}
    plot_sle(x_test,predicted_data,results.loc[idx,'X#1'],results.loc[idx,'T_EP'],
             results.loc[idx,'Component#1'],results.loc[idx,'Component#2'],experimental[(smiles_1, smiles_2)])
results.to_csv('../results/EP_prediction.csv',index=False)
