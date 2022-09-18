''' Read gamma inf files '''
#%%Imports
import pandas as pd
import re, os

#%% Functions
def read_out(df, path='../data/gamma_inf'):
    data_InfD = []
    cols = {'mu':'Chemical potential','partial_pressure':'Log10','Free_energy_COSMO':'Free energy','H_int':'Total mean','H_MF':'Misfit', 'H_HB': 'H-Bond', 'H_vdW': 'VdW', 'ln_gamma': 'Ln'}
    header = []
    for idx in df.index:
        InfD = []
        inchi_1 = df.loc[idx, 'inchi#1']
        inchi_2 = df.loc[idx, 'inchi#2']
        path_idx = f'{path}/{inchi_1}_{inchi_2}.out'
        if os.path.exists(path_idx):
            with open(path_idx, 'r') as inpf:
                text = inpf.read()
            for head, col in cols.items():
                if f'{head}_InfD#1' not in header:
                    header.append(f'{head}_InfD#1')
                if f'{head}_InfD#2' not in header:
                    header.append(f'{head}_InfD#2')
                data = re.findall(rf'{col}[^\d|-]+(.*)', text)
                if head != 'ln_gamma' and len(data) == 8:
                    InfD.append(data[6].split(' ')[0])
                    InfD.append(data[3].split(' ')[0])
                elif head == 'ln_gamma' and len(data) == 4:
                    InfD.append(data[2].split(' ')[0])
                    InfD.append(data[1].split(' ')[0])
                else:
                    print('Error: ', idx, ' ', head)
            data_InfD.append(InfD)
    results = pd.concat([df[['Component#1', 'Smiles#1', 'Component#2', 'Smiles#2']], pd.DataFrame(data=data_InfD, columns = header)], axis=1)
    return results

#%% Read mu in mixture
df_init = pd.read_csv('../data/DES_init_update.csv').drop_duplicates(subset=['Smiles#1','Smiles#2']).reset_index(drop=True) 
df_1 = read_out(df_init)
df_1.to_csv('../descriptors/mixture/infinite_dilution.csv', index=False)
