''' Add solubility and bins'''
import pandas as pd
import numpy as np

cols = ['Component#1','Component#2','Smiles#1','Smiles#2','X#1','T_EP','PD','Type']
df = pd.read_csv('../data/DES_init_update.csv')[cols]
df_unique = df.drop_duplicates(subset=['Smiles#1','Smiles#2']).reset_index(drop=True)
df_groups = {(df_unique.loc[idx,'Smiles#1'],df_unique.loc[idx,'Smiles#2']):idx for idx in df_unique.index}
for idx in df.index:
    df.loc[idx,'groups'] = df_groups[(df.loc[idx,'Smiles#1'],df.loc[idx,'Smiles#2'])]

x = [i/20 for i in range(2,19)]
for idx in df.index:
    x1 = df.loc[idx,'X#1']
    dif = [abs(x_i-x1) for x_i in x]
    df.loc[idx,'X#1'] = x[dif.index(np.min(dif))]
    
df = df.sort_values(by = ['T_EP']).drop_duplicates(subset = ['Smiles#1', 'Smiles#2','X#1'], keep = 'first').sort_index()

df.to_csv('../descriptors/mixture/main.csv',index=False)
