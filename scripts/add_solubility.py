''' Add solubility and bins'''
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

cols = ['Component#1','Component#2','Smiles#1','Smiles#2','X#1','T_EP','Phase_diagram']
df = pd.read_csv('../data/DES_init_update.csv')[cols]
for idx in df.index:
    smiles_1 = df.loc[idx, 'Smiles#1']
    smiles_2 = df.loc[idx, 'Smiles#2']
    x1 = df.loc[idx, 'X#1']
    x2 = 1-x1
    MolWt1=Descriptors.MolWt(Chem.AddHs(Chem.MolFromSmiles(smiles_1)))
    MolWt2=Descriptors.MolWt(Chem.AddHs(Chem.MolFromSmiles(smiles_2)))
    df.loc[idx, 'Solubility#1'] = 100*MolWt1*x1/(MolWt2*x2)
    df.loc[idx, 'Solubility#2'] = 100*MolWt2*x2/(MolWt1*x1)
df['bins'] = np.digitize(df['X#1'], bins=[0]+[round(i,2) for i in np.linspace(0.15,0.85,8)]+[1])
df.to_csv('../descriptors/mixture/main.csv',index=False)