
import pandas as pd
import numpy as np

#%% Default functions
paths = {
 'mu_05': '../descriptors/mixture/mu_05.csv',
 'mu_inf': '../descriptors/mixture/mu_infinite_dilution.csv',
 'clusterig': '../descriptors/compounds/calculated/clusterization.csv',
 '2D': '../descriptors/compounds/calculated/2D_descriptors.csv',
 'mom': '../descriptors/compounds/calculated/sigma_moments.csv',
 'profile': '../descriptors/compounds/calculated/sigma_profiles.csv',
 'potent': '../descriptors/compounds/calculated/sigma_potential.csv',
 'thermo': '../descriptors/compounds/measured/thermochem.csv'
 }

def add_separate(columns, desc1, desc2):
    desc1 = desc1.rename(columns={col:f'{col}#1' for col in columns})
    desc2 = desc2.rename(columns={col:f'{col}#2' for col in columns})
    desc = pd.concat([desc1, desc2], axis=1)
    return desc

def add_compound_descriptors(df, df_desc, function=None, columns_f=[]):
    columns = df_desc.drop(['Component','Smiles'], axis=1).columns
    if columns_f == 'all':
        columns_f = columns
    results = pd.DataFrame()
    results[df.columns]=df[df.columns]
    for idx in results.index:
        smiles_1 = results.loc[idx,'Smiles#1']
        smiles_2 = results.loc[idx,'Smiles#2']
        if smiles_2 == 'C[N+](C)(C)CC(=O)[O-].O':
            smiles_2 = 'C[N+](C)(C)CC(=O)[O-]' 
        desc_1 = df_desc[df_desc['Smiles']==smiles_1].drop(['Component','Smiles'],axis=1).reset_index(drop=True)
        desc_2 = df_desc[df_desc['Smiles']==smiles_2].drop(['Component','Smiles'],axis=1).reset_index(drop=True)
        descs = add_separate(columns,desc_1,desc_2)
        x1 = results.loc[idx, 'X#1']
        for col in columns:
            if col not in columns_f:
                results.loc[idx, [f'{col}#1',f'{col}#2']] = descs.loc[0,[f'{col}#1',f'{col}#2']] 
        for col in columns_f:
            results.loc[idx, col] = function(col,descs,x1).iloc[0]
    return results

def add_mixture_descriptors(df, df_desc, function=None, columns_f=[]):
    columns = []
    for col in df_desc.drop(['Component#1','Smiles#1','Component#2','Smiles#2'], axis=1).columns:
        col = col.split('#')[0]
        if col not in columns:
            columns.append(col)
    if columns_f == 'all':
        columns_f = columns
    results = pd.DataFrame()
    results[df.columns]=df[df.columns]
    for idx in results.index:
        smiles_1 = results.loc[idx,'Smiles#1']
        smiles_2 = results.loc[idx,'Smiles#2']
        descs = df_desc[(df_desc['Smiles#1']==smiles_1) & (df_desc['Smiles#2']==smiles_2)].drop(['Component#1','Smiles#1','Component#2','Smiles#2'],axis=1).reset_index(drop=True)
        x1 = results.loc[idx, 'X#1']
        for col in columns:
            if col not in columns_f:
                results.loc[idx, [f'{col}#1',f'{col}#2']] = descs.loc[0,[f'{col}#1',f'{col}#2']] 
        for col in columns_f:
            results.loc[idx, col] = function(col,descs,x1).iloc[0]
    return results

#%% Functions for merging    
def log_ratio(col,df,x1):
    x2=1-x1
    return df[f'{col}#1']*np.log(x1)+df[f'{col}#2']*np.log(x2)

def ratio(col,df,x1):
    x2=1-x1
    return df[f'{col}#1']*x1+df[f'{col}#2']*x2

def deg(col,df,x1):
    x2=1-x1
    return np.sqrt(((df[f'{col}#1']**2)**x1)*((df[f'{col}#2']**2)**x2))

#%% Class descriptors
class descriptors:
    def __init__(self, paths=paths):
        self.mu_05 = pd.read_csv(paths['mu_05'])
        self.mu_inf = pd.read_csv(paths['mu_inf'])
        self.cluster = pd.read_csv(paths['clusterig']).rename(columns={'cluster_birch':'cluster'})
        self.rdkit_2D = pd.read_csv(paths['2D'])
        self.mom = pd.read_csv(paths['mom'])
        self.profile = pd.read_csv(paths['profile'])
        self.potent = pd.read_csv(paths['potent'])
        self.thermo = pd.read_csv(paths['thermo']).drop(['inchi'], axis=1)
    
    def add_cluster(self, df):
        cluster = pd.get_dummies(self.cluster,columns=['cluster'])
        results = pd.DataFrame()
        results[df.columns]=df[df.columns]
        for idx in results.index:
            smiles_1 = results.loc[idx,'Smiles#1']
            smiles_2 = results.loc[idx,'Smiles#2']
            descs_1 = cluster[cluster['Smiles']==smiles_1].drop(['Component','Smiles'],axis=1).reset_index(drop=True)
            descs_2 = cluster[cluster['Smiles']==smiles_2].drop(['Component','Smiles'],axis=1).reset_index(drop=True)
            for col in descs_1.columns:
                results.loc[idx, col] = descs_1.loc[0, col]+descs_2.loc[0, col]
        return results
        
    def add_2D(self, df, function=ratio, columns_f=['NumValenceElectrons','NumAliphaticCarbocycles',
                                                    'NumAliphaticHeterocycles','NumAromaticCarbocycles',
                                                    'NumAromaticHeterocycles', 'NumHAcceptors',
                                                    'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds',
                                                    'fr_Al_COO', 'fr_Al_OH', 'fr_Ar_COO','fr_Ar_N', 
                                                    'fr_Ar_OH', 'fr_C_O_noCOO', 'fr_NH0', 'fr_NH1', 
                                                    'fr_NH2','fr_Ndealkylation1', 'fr_amide',
                                                    'fr_halogen', 'fr_phenol', 'fr_quatN']):
        return add_compound_descriptors(df, self.rdkit_2D, function=function, columns_f=columns_f)
    
    def add_mom(self, df, function=None, columns_f=[]):
        return add_compound_descriptors(df, self.mom, function=function, columns_f=columns_f)
    
    def add_profile(self, df, function=ratio, columns_f='all'):
        return add_compound_descriptors(df, self.profile, function=function, columns_f=columns_f)
    
    def add_potent(self, df, function=ratio, columns_f='all'):
        return add_compound_descriptors(df, self.potent, function=function, columns_f=columns_f)
    
    def add_thermo(self, df, function=None, columns_f=[]):
        return add_compound_descriptors(df, self.thermo, function=function, columns_f=columns_f)
    
    def add_mu_05(self, df, function=deg, columns_f='all'):
        return add_mixture_descriptors(df, self.mu_05, function=function, columns_f=columns_f)
    
    def add_mu_inf(self, df, function=deg, columns_f='all'):
        return add_mixture_descriptors(df, self.mu_inf, function=function, columns_f=columns_f)
    
    def add_several(self, df, thermo=True, cluster=True, rdkit_2D=True, mom=True, mu_05=True, mu_inf=True, profile=True, potential=True):
        df_results = pd.DataFrame()
        df_results[df.columns]=df[df.columns]
        if thermo:
            df_results = self.add_thermo(df_results)
        if cluster:
            df_results = self.add_cluster(df_results)
        if rdkit_2D:
            df_results = self.add_2D(df_results)
        if mom:
            df_results = self.add_mom(df_results)
        if mu_05:
            df_results = self.add_mu_05(df_results)
        if mu_inf:
            df_results = self.add_mu_inf(df_results)
        if profile:
            df_results = self.add_profile(df_results)
        if potential:
            df_results = self.add_potent(df_results)
        return df_results
            
            
    

            
                
            
