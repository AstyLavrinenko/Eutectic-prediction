''' add Kf '''
import pandas as pd

cols = ['Component', 'Smiles', 'inchi', 'H', 'T']
df = pd.read_csv('../data/DES_init_update.csv')
compound_1 = df[[f'{col}#1' for col in cols]].rename(columns = {f'{col}#1': col for col in cols})
compound_2 = df[[f'{col}#2' for col in cols]].rename(columns = {f'{col}#2': col for col in cols})
compound = pd.concat([compound_1, compound_2]).drop_duplicates('Smiles')
R = 8.31446261815324
for idx in compound.index:
    compound.loc[idx, 'Kf'] = R*compound.loc[idx, 'T']**2/(1000*compound.loc[idx, 'H'])
compound.to_csv('../descriptors/compounds/measured/thermochem.csv', index=False)
