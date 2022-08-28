''' Read mu mixture files '''
#%%Imports
import pandas as pd
import re, os

#%% Functions
def find_in_file(text, name, idxes=[0], slices=[0,2,3,7]):
    data_final = []
    for idx in idxes:
        data =  re.findall(rf'{name}\s+(.+)$', text, flags=re.MULTILINE)[idx]
        data = re.split('\s+', data)
        data_slised = []
        for idx_sli in range(0, len(slices),2):
            data_slised += data[slices[idx_sli]:slices[idx_sli+1]]
        data_final.append(data_slised)
    return data_final

def read_mu(path, df, num):
    data_final = []
    for idx in df.index:
        inchi_1 = df.loc[idx, 'inchi#1']
        inchi_2 = df.loc[idx, 'inchi#2']
        path_idx = f'{path}/{inchi_1}_{inchi_2}.tab'
        if os.path.exists(path_idx):
            with open(path_idx, 'r') as inpf:
                text = inpf.read()
            header = find_in_file(text, 'Nr Compound')[0]
            if num == 2:
                segments_1 = find_in_file(text, f'{inchi_1}',[0,3])
                segments_2 = find_in_file(text, f'{inchi_2}',[0,3])
                data=segments_1[0]+segments_2[1]+segments_1[1]+segments_2[0]
                columns=[f'{head}_1#1' for head in header]+[f'{head}_1#2' for head in header]+[f'{head}_0#1' for head in header]+[f'{head}_0#2' for head in header]
            elif num == 1:
                segments_1 = find_in_file(text, f'{inchi_1}')
                segments_2 = find_in_file(text, f'{inchi_2}')
                data=segments_1[0]+segments_2[0]
                columns=[f'{head}_05#1' for head in header]+[f'{head}_05#2' for head in header]
            else:
                return print('num should be 1 or 2')
            data_final.append(data)
        else:
            print('path does not exist: ', path_idx)
            data_final.append([])
    df_result = pd.DataFrame(data=data_final, columns=columns)
    df_result = pd.concat([df[['Component#1','Smiles#1','Component#2','Smiles#2']],df_result], axis=1)
    return df_result

#%% Read mu in mixture
df_init = pd.read_csv('../data/DES_init_update.csv').drop_duplicates(subset=['Smiles#1','Smiles#2']).reset_index(drop=True) 
path_1 = '../data/mu_infinite_dilution'
path_2 = '../data/mu_05/out_files'
df_1 = read_mu(path_1, df_init, 2)
df_1.to_csv('../descriptors/mixture/mu_infinite_dilution.csv', index=False)
df_2 = read_mu(path_2, df_init, 1)
df_2.to_csv('../descriptors/mixture/mu_05.csv', index=False)
