# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 13:43:54 2022

@author: computer
"""
fig, ax = plt.subplots(figsize=(6,6))
ax=sns.barplot(data=results,y='r2_val',x='model_name')
ax.set_xlabel('ML model',{'fontsize':14})
ax.set_ylabel('R2',{'fontsize':14})
plt.savefig('../results/r2.eps',dpi=1200,bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(6,6))
ax=sns.barplot(data=results,y='rmse_val',x='model_name')
ax.set_xlabel('ML model',{'fontsize':14})
ax.set_ylabel('RMSE',{'fontsize':14})
plt.savefig('../results/rmse.eps',dpi=1200,bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(6,6))
ax=sns.barplot(data=results,y='mape_val',x='model_name')
ax.set_xlabel('ML model',{'fontsize':14})
ax.set_ylabel('MAPE',{'fontsize':14})
plt.savefig('../results/mape.eps',dpi=1200,bbox_inches='tight')
plt.show()

#%%
df_cosmo = pd.read_csv('../results/SLE_COSMO.csv')
df_ml = pd.read_csv('../results/EP_prediction.csv')

for key in ['CA17','C+A17','CA19','C+A19']:
    print(df_cosmo.groupby(by=[f'mark_{key}'])['Type'].count())

for key in ['CA17','C+A17','CA19','C+A19']:
    df_new = df_cosmo[(df_cosmo[f'mark_{key}']=='1')&(df_cosmo['PD']=='Yes')]
    print(key)
    print(evaluate_model(np.array(df_new['T_EP']), 
                     np.array(df_new[f'T_{key}'])))
    print(evaluate_model(np.array(df_new['X#1']), 
                     np.array(df_new[f'X_{key}'])))
for key in ['GBR','RFR','SVR','KNN']:
    df_new = df_ml
    print(key)
    print(evaluate_model(np.array(df_new['T_EP']), 
                     np.array(df_new[f'T_{key}'])))
    print(evaluate_model(np.array(df_new['X#1']), 
                     np.array(df_new[f'X_{key}'])))
    
#%%
data_new = pd.DataFrame(columns=['mark','model'])
data_new['mark'] = pd.concat([df_cosmo['mark_CA17'].rename({'mark_CA17':'mark'}),df_cosmo['mark_C+A17'].rename({'mark_C+A17':'mark'})])
data_new['model'] = ['CA' for i in range(237)]+['C+A' for i in range(237)]
data_new=data_new.reset_index(drop=True)
data_new.loc[data_new[data_new['mark']=='2'].index,'mark'] = 'multi'
data_new.loc[data_new[data_new['mark']=='3'].index,'mark'] = 'multi'
fig, ax = plt.subplots()
ax=sns.histplot(data=data_new,x='model',hue='mark',hue_order=['0','1','multi','no H data','Error'],color=sns.color_palette('pastel'),multiple='dodge',shrink=0.8,stat='count')
legend = ax.get_legend()
handles = legend.legendHandles
legend.remove()
ax.legend(handles, ['without intersection', 'single intersection', 'multiple intersections', 'without enthalpy data','can not be calculated by COSMO-RS'],bbox_to_anchor=(1.04,0.9), loc='lower center', borderaxespad=0)
plt.savefig('../results/type_cosmo.eps',dpi=1200,bbox_inches='tight')
plt.show()

#%%
df_new = df_cosmo[(df_cosmo['mark_CA17']=='1')&(df_cosmo['mark_C+A17']=='1')]
plt.plot(np.arange(170,511,31),np.arange(170,511,31),alpha = 0.6,color='green',linewidth = 2)
plt.scatter(df_new['T_EP'],pd.to_numeric(df_new['T_C+A17']),s=50,facecolors='none',edgecolors='orange',linewidth = 2,label='C+A')
plt.scatter(df_new['T_EP'],pd.to_numeric(df_new['T_CA17']),s=50,facecolors='none',edgecolors='purple',linewidth = 2,label='CA')
plt.xlim(170,480)
plt.ylim(170,480)
plt.legend()
plt.xlabel('Experimental temperature, K')
plt.ylabel('Predicted temperature, K')
plt.savefig('../results/CA17_vs_C+A17.eps',dpi=1200,bbox_inches='tight')
plt.show()

#%%
import read_sle_files

temps=[]
concs_1=[]
concs_2=[]
names_graph=[]

idx = [635,517,265,877]
for idx in [635,517,265,877]:
    names_1 = df_unique.loc[idx,'Component#1']
    names_2 = df_unique.loc[idx,'Component#2']
    inchi_1 = df_unique.loc[idx,'inchi#1']
    inchi_2 = df_unique.loc[idx,'inchi#2']
    names = f'{names_1}#{names_2}'
    inchi = f'{inchi_1}_{inchi_2}'
    path_inp = '../data/SLE_COSMO/SLE_CA_17'
    path = f'{path_inp}/{inchi}'
    temp, conc_1, conc_2 = read_sle_files.read_SLE_tab(path, path_2=None)
    temps.append(temp)
    concs_1.append(conc_1)
    concs_2.append(conc_2)
    names_graph.append(names_1)

figure, axis = plt.subplots(2, 2,figsize=(12,8))
axis[0, 0].plot(concs_2[0], temps[0], '-or', linewidth = 1, markersize = 3)
axis[0, 0].plot(concs_1[0], temps[0], '-ob',linewidth = 1, markersize = 3)

axis[0, 1].plot(concs_2[1], temps[1], '-or',linewidth = 1, markersize = 3)
axis[0, 1].plot(concs_1[1], temps[1], '-ob',linewidth = 1, markersize = 3)

axis[1, 0].plot(concs_2[2], temps[2], '-or',linewidth = 1, markersize = 3)
axis[1, 0].plot(concs_1[2], temps[2], '-ob',linewidth = 1, markersize = 3)

axis[1, 1].plot(concs_2[3], temps[3], '-or',linewidth = 1, markersize = 3)
axis[1, 1].plot(concs_1[3], temps[3], '-ob',linewidth = 1, markersize = 3)

for i,ax in enumerate(axis.flat):
    ax.set_xlabel(f"Mole fraction of {names_graph[i]}",{'fontsize': 10})
    ax.set_ylabel('Temperature, K',{'fontsize': 10})
    ax.set_ylim(30,630)
    ax.set_yticks(np.arange(30, 680, 100),fontsize=10)
    ax.set_xticks(np.arange(0, 1.02, 0.2),fontsize=10)
plt.savefig('../results/Representative_sle.eps',dpi=1200,bbox_inches='tight')
plt.show()

#%%
import read_sle_files
#from EP_prediction import fit_model
import add_descriptors

df = pd.read_csv('../descriptors/mixture/main.csv')
df = add_descriptors.descriptors().add_selected_features(df)
to_drop = ['Component#1','Smiles#1','Component#2','Smiles#2','T_EP','PD','groups']
df = pd.get_dummies(df, columns = ['Type'])

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
concs = [i/20 for i in range(2,19)]

temps=[]
concs_1=[]
concs_2=[]
names_graph=[]
y_preds = []
exp_data = []

for idx in [63,441,742,1431]:
    smiles_1 = df_unique.loc[idx, 'Smiles#1']
    smiles_2 = df_unique.loc[idx, 'Smiles#2']
    names_1 = df_unique.loc[idx,'Component#1']
    names_2 = df_unique.loc[idx,'Component#2']
    inchi_1 = df2.loc[df2[(df2['Smiles#1']==smiles_1)&(df2['Smiles#2']==smiles_2)&(df2['T_EP']==df_unique.loc[idx,'T_EP'])].index[0],'inchi#1']
    inchi_2 = df2.loc[df2[(df2['Smiles#1']==smiles_1)&(df2['Smiles#2']==smiles_2)&(df2['T_EP']==df_unique.loc[idx,'T_EP'])].index[0],'inchi#2']
    names = f'{names_1}#{names_2}'
    inchi = f'{inchi_1}_{inchi_2}'
    path_inp = '../data/SLE_COSMO/SLE_CA_17'
    path = f'{path_inp}/{inchi}'
    temp, conc_1, conc_2 = read_sle_files.read_SLE_tab(path, path_2=None)
    temps.append(temp)
    concs_1.append(conc_1)
    concs_2.append(conc_2)
    names_graph.append(names_1)
    
    y_pred_dict = {}
    
    exp_data.append(experimental[(smiles_1, smiles_2)])
    x_test = pd.DataFrame(columns = ['Component#1', 'Component#2', 'PD','Type','groups', 'T_EP'])
    x_test['X#1'] = concs
    x_test['Smiles#1'] = smiles_1
    x_test['Smiles#2'] = smiles_2
    x_type = df.loc[idx,['Type_IL mixture','Type_Type III','Type_Type V']]
    x_test = add_descriptors.descriptors().add_selected_features(x_test).drop(['Type'],axis=1)
    x_test[['Type_IL mixture','Type_Type III','Type_Type V']] = x_type
    x_train = df.drop(index=df[(df['Smiles#1']==smiles_1) & (df['Smiles#2']==smiles_2)].index)
    y_train = df.drop(index=df[(df['Smiles#1']==smiles_1) & (df['Smiles#2']==smiles_2)].index)['T_EP']
    for model in ['GBR','RFR','SVR','KNN']:
        y_pred_dict[model] = fit_model(x_train.drop(to_drop,axis=1),y_train,x_test.drop(to_drop,axis=1),model)
    y_preds.append(y_pred_dict)




figure, axis = plt.subplots(2, 2,figsize=(12,9))
axis[0, 0].plot(concs_2[0], temps[0], color='gray', linewidth = 2, label='COSMO-RS')
axis[0, 0].plot(concs_1[0], temps[0], color='gray', linewidth = 2)
axis[0, 0].plot(exp_data[0]['X#1'], exp_data[0]['T_EP'],alpha = 0.6, linestyle = '--', marker = 'o', markersize = 8, color='green', linewidth = 2, label='Experimental data')
axis[0, 0].scatter(concs, y_preds[0]['GBR'], s=50, facecolors='none', edgecolors='blueviolet', linewidth = 2, label='GBR')
axis[0, 0].scatter(concs, y_preds[0]['RFR'], s=50, facecolors='none', edgecolors='gold', linewidth = 2, label='RFR')
axis[0, 0].scatter(concs, y_preds[0]['SVR'], s=50, facecolors='none', edgecolors='maroon', linewidth = 2, label='SVR')
axis[0, 0].scatter(concs, y_preds[0]['KNN'], s=50, facecolors='none', edgecolors='salmon', linewidth = 2, label='KNN')

axis[0, 1].plot(concs_2[1], temps[1], color='gray', linewidth = 2, label='COSMO-RS')
axis[0, 1].plot(concs_1[1], temps[1], color='gray', linewidth = 2)
axis[0, 1].plot(exp_data[1]['X#1'], exp_data[1]['T_EP'],alpha = 0.6, linestyle = '--', marker = 'o', markersize = 8, color='green', linewidth =2, label='Experimental data')
axis[0, 1].scatter(concs, y_preds[1]['GBR'], s=50, facecolors='none', edgecolors='blueviolet', linewidth = 2, label='GBR')
axis[0, 1].scatter(concs, y_preds[1]['RFR'], s=50, facecolors='none', edgecolors='gold', linewidth = 2, label='RFR')
axis[0, 1].scatter(concs, y_preds[1]['SVR'], s=50, facecolors='none', edgecolors='maroon', linewidth = 2, label='SVR')
axis[0, 1].scatter(concs, y_preds[1]['KNN'], s=50, facecolors='none', edgecolors='salmon', linewidth = 2, label='KNN')

axis[1, 0].plot(concs_2[2], temps[2], color='gray' ,linewidth = 2, label='COSMO-RS')
axis[1, 0].plot(concs_1[2], temps[2], color='gray' ,linewidth = 2)
axis[1, 0].plot(exp_data[2]['X#1'], exp_data[2]['T_EP'],alpha = 0.6, linestyle = '--', marker = 'o', markersize = 8, color='green', linewidth = 2, label='Experimental data')
axis[1, 0].scatter(concs, y_preds[2]['GBR'], s=50, facecolors='none', edgecolors='blueviolet', linewidth = 2, label='GBR')
axis[1, 0].scatter(concs, y_preds[2]['RFR'], s=50, facecolors='none', edgecolors='gold', linewidth = 2, label='RFR')
axis[1, 0].scatter(concs, y_preds[2]['SVR'], s=50, facecolors='none', edgecolors='maroon', linewidth = 2, label='SVR')
axis[1, 0].scatter(concs, y_preds[2]['KNN'], s=50, facecolors='none', edgecolors='salmon', linewidth = 2, label='KNN')

axis[1, 1].plot(exp_data[3]['X#1'], exp_data[3]['T_EP'], alpha = 0.6,linestyle = '--', marker = 'o', markersize = 8, color='green', linewidth = 2, label='Experimental data')
axis[1, 1].scatter(concs, y_preds[3]['GBR'], s=50, facecolors='none', edgecolors='blueviolet', linewidth = 2, label='GBR')
axis[1, 1].scatter(concs, y_preds[3]['RFR'], s=50, facecolors='none', edgecolors='gold', linewidth = 2, label='RFR')
axis[1, 1].scatter(concs, y_preds[3]['SVR'], s=50, facecolors='none', edgecolors='maroon', linewidth = 2, label='SVR')
axis[1, 1].scatter(concs, y_preds[3]['KNN'], s=50, facecolors='none', edgecolors='salmon', linewidth = 2, label='KNN')

lims = [(310,605),(190,410),(265,465),(255,490)]
for i,ax in enumerate(axis.flat):
    ax.set_xlabel(f"Mole fraction of {names_graph[i]}",{'fontsize': 12})
    ax.set_ylabel('Temperature, K',{'fontsize': 12})
    ax.set_ylim(lims[i])
    #ax.set_yticks(np.arange(250, 600, 100),fontsize= 8)
    ax.set_xticks(np.arange(0, 1.02, 0.2),fontsize= 8)
#plt.savefig('../results/Representative_sle.eps',dpi=1200,bbox_inches='tight')
axis[0, 1].legend(fontsize = 12, bbox_to_anchor=(1.04,0.9), loc='upper left', borderaxespad=0)
plt.savefig('../results/Representative_sle_ml.eps',dpi=1200,bbox_inches='tight')
plt.show()

#%%

features = [
'X#1',
'T#1',
'T#2',
'NPR1#2',
'NPR2#2',
'RadiusOfGyration#2',
'InertialShapeFactor#2',
'SpherocityIndex#1',
'SpherocityIndex#2',
'MolWeight#2',
'WAPS#1',
'WAPS#2',
'WANS#2',
'ln_gamma_InfD#1',
'ln_gamma_InfD#2',
's_profile_2#ratio',
's_profile_3#ratio',
's_profile_4#ratio',
's_profile_5#ratio',
's_profile_6#ratio'
    ]

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import numpy as np
 
# import file with data
data = x_train[features]
mask = np.triu(np.ones_like(data.corr()))
fig, ax = plt.subplots(figsize=(15,15))
ax = sb.heatmap(data.corr(), cmap="YlGnBu", annot=True, mask=mask)
plt.savefig('../results/corr_2.eps',dpi=1200,bbox_inches='tight')
plt.show()

#%%
fig, ax = plt.subplots(figsize=(7,4))
ax = sns.boxplot(x='Type', y='T_EP', data=df, hue='PD')
ax, test_results = add_stat_annotation(ax, x='Type', y='T_EP', data=df, hue='PD',
                                   box_pairs=[(('Type III','No'), ('Type III','Yes')), (('Type V','No'), ('Type V','Yes')), (('IL mixture','Yes'), ('Type III','Yes')), (('IL mixture','Yes'), ('Type V','Yes')), (('Type III','Yes'), ('Type V','Yes')), (('Type III','No'), ('Type V','No'))],
                                   test='Mann-Whitney', text_format='star', loc='outside', verbose=2)
ax.set_ylabel('Temperature, K')
plt.savefig('../results/stat.eps',dpi=1200,bbox_inches='tight')
plt.show()