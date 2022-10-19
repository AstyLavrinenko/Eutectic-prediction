''' read sigma moments '''
#%% Imports
import re
import pandas as pd

#%% Functions
def read_mom(inchies):
    path_inp = '../data/sigma_moments_17/out_files'
    desc_moments = []
    for inchi in inchies:
        with open(f'{path_inp}/{inchi}.mom', 'r') as inpf:
            text = inpf.read()
        header = re.findall(r'Molecule\s+(.*)\s+', text)
        header = re.split('\s+', header[0])[:28]
        mom = re.findall(rf'{inchi}\s+(.*)\s+', text)
        mom = [float(number) for number in re.split('\s+', mom[0])]
        mom.append(mom[header.index('charge_p')]/mom[header.index('area_p')])
        mom.append(mom[header.index('charge_n')]/mom[header.index('area_n')])
        header+=['WAPS','WANS']
        desc_moments.append(mom[:1]+mom[2:7]+mom[15:18]+mom[19:26]+mom[28:])
        header = header[:1]+header[2:7]+header[15:18]+header[19:26]+header[28:]
    df = pd.DataFrame(data=desc_moments, columns = header)
    return df

def read_out(inchies):
    path_inp = '../data/sigma_moments_17/out_files'
    desc_moments = []
    cols = {'E_COSMO':'E_COSMO-E_gas','COSMO_charge':'Total COSMO charge','Dipole_moment':'Dipole moment','Hb_acc':'accept','Hb_don':'donor'}
    header = list(cols.keys())
    for inchi in inchies:
        with open(f'{path_inp}/{inchi}.out', 'r') as inpf:
            text = inpf.read()
        mom = [re.findall(rf'{col}[^\d|-]+(.*)', text)[0].split(' ')[0] for col in cols.values()]
        desc_moments.append(mom)
    df = pd.DataFrame(data=desc_moments, columns = header)
    return df

#%% Main code
df_init = pd.read_csv('../descriptors/compounds/measured/thermochem.csv')
mom_mom = read_mom(df_init['inchi'])
mom_out = read_out(df_init['inchi'])
df_resulted = pd.concat([df_init[['Component', 'Smiles']],mom_mom,mom_out],axis=1)
df_resulted.to_csv('../descriptors/compounds/calculated/sigma_moments.csv', index=False)
