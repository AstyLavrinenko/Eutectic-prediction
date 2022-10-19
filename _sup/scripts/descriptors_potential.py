''' Read and integrate sigma-potential by COSMO '''
#%% Imports
import re, io
import pandas as pd
from scipy import integrate

#%% Functions
def integrate_sigma(df, grids):
    integrals = [0.0]*(len(grids) - 1)
    for i in range(len(integrals)):
        left = grids[i]
        right = grids[i+1]
        p = df.p[(df.sigma >= left) & (df.sigma < right)]
        sigma = df.sigma[(df.sigma >= left) & (df.sigma < right)]
        integrals[i] = integrate.simpson(p, sigma)
    return integrals

def read_sigma(inchies, grids=[-0.025,-0.01,0.01,0.025]):
    path_inp = '../data/sigma_potential_17/out_files'
    sigmas = []
    for inchi in inchies:
        with open(f'{path_inp}/{inchi}.tab', 'r') as inpf:
            text = inpf.read()
        segments = re.search(r'mu\(1\)[^\n]+\n(.+)$', text, re.DOTALL).group(1).rstrip()
        df_sigma = pd.read_csv(io.StringIO(segments), names = ['sigma','p'], sep = r'\s+', engine = 'python')
        df_sigma.drop([0], inplace = True)
        df_sigma = df_sigma.astype('float')
        sigmas.append(integrate_sigma(df_sigma, grids))
    df = pd.DataFrame(data=sigmas, columns=['s_potential_1','s_potential_2','s_potential_3'])
    return df

#%% Main code
df_init = pd.read_csv('../descriptors/compounds/measured/thermochem.csv')
desc_sigma = read_sigma(df_init['inchi'])
df_resulted = pd.concat([df_init[['Component', 'Smiles']],desc_sigma],axis=1)
df_resulted.to_csv('../descriptors/compounds/calculated/sigma_potential.csv', index=False)
