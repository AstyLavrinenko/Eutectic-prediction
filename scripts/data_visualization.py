''' Removing the outlets and statistic '''

import pandas as pd
import numpy as np
import seaborn as sns
from statannot import add_stat_annotation
from rdkit import Chem
from rdkit.Chem import Descriptors

#%% Prepare data
df = pd.read_csv('../data/DES_init_update.csv')
to_drop = ['T_EP','X#1','Component#1','Smiles#1','Component#2','Smiles#2']
#%% Add thermo features
R = 8.31446261815324
for idx in df.index:
    smiles_1 = df.loc[idx, 'Smiles#1']
    smiles_2 = df.loc[idx, 'Smiles#2']
    X_1 = df.loc[idx, 'X#1']
    MolWt1=Descriptors.MolWt(Chem.MolFromSmiles(smiles_1))
    MolWt2=Descriptors.MolWt(Chem.MolFromSmiles(smiles_2))
    df.loc[idx, 'Solubility#1'] = 100*MolWt1*X_1/(MolWt2*(1-X_1))
    df.loc[idx, 'Solubility#2'] = 100*MolWt2*(1-X_1)/(MolWt2*X_1)
    df.loc[idx, 'Kf#1'] = R*df.loc[idx, 'T#1']**2/(1000*df.loc[idx, 'H#1'])
    df.loc[idx, 'Kf#2'] = R*df.loc[idx, 'T#2']**2/(1000*df.loc[idx, 'H#2'])
df.info()
#%% outliers
Q1, Q3 = np.percentile(df['T_EP'], [25,75])
IQR = Q3 - Q1
outliers = np.where((df['T_EP'] > (Q3+1.5*IQR)) | (df['T_EP'] <= (Q1-1.5*IQR)))
df_out = df.iloc[outliers[0]]
#df.drop(outliers[0], inplace = True)
#%%
sns.boxplot(df['T_EP'])
ax = sns.boxplot(x='Type', y='T_EP', data=df, hue='Phase_diagram')
ax, test_results = add_stat_annotation(ax, x='Type', y='T_EP', data=df, hue='Phase_diagram',
                                   box_pairs=[(('III','No'), ('III','Yes')), (('V','No'), ('V','Yes')), (('I','Yes'), ('III','Yes')), (('I','Yes'), ('V','Yes')), (('III','Yes'), ('V','Yes')), (('III','No'), ('V','No'))],
                                   test='Mann-Whitney', text_format='star', loc='outside', verbose=2)
sns.boxplot(df['X#1'])
ax = sns.boxplot(x='Type', y='X#1', data=df, hue='Phase_diagram')
ax, test_results = add_stat_annotation(ax, x='Type', y='X#1', data=df, hue='Phase_diagram',
                                   box_pairs=[(('III','No'), ('III','Yes')), (('V','No'), ('V','Yes')), (('I','Yes'), ('III','Yes')), (('I','Yes'), ('V','Yes')), (('III','Yes'), ('V','Yes')), (('III','No'), ('V','No'))],
                                   test='Mann-Whitney', text_format='star', loc='outside', verbose=2)
df['bins'] = np.digitize(df['X#1'], bins=np.linspace(0,1,21))

df.to_csv('../descriptors/a_thermochem.csv', index=False)
