''' Plot SLE data '''

import pandas as pd
import read_sle_files

df=pd.read_csv('../data/DES_init_update.csv').sort_values(by = ['T_EP']).drop_duplicates(subset = ['Smiles#1', 'Smiles#2'], keep = 'first').sort_index()
keys = ['CA17','C+A17','CA19','C+A19']
for key in keys:
    results = read_sle_files.DES(df.dropna(subset=['H#1','H#2'],axis=0),key).plot()
    df=pd.concat([df,results], axis=1)

