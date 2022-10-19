
import pandas as pd
import numpy as np

#%% Default functions
paths = {
 'InfDilut': '../descriptors/mixture/infinite_dilution.csv',
 'rdkit': '../descriptors/compounds/calculated/rdkit_descriptors.csv',
 'mom': '../descriptors/compounds/calculated/sigma_moments.csv',
 'profile': '../descriptors/compounds/calculated/sigma_profiles.csv',
 'thermo': '../descriptors/compounds/measured/thermochem.csv'
 }
#%% Functions for merging    
def ratio(col,df,x1):
    x2=1-x1
    name = 'ratio'
    return name, df[f'{col}#1']*x1+df[f'{col}#2']*x2  

def add_separate(columns, desc1, desc2):
    desc1 = desc1.rename(columns={col:f'{col}#1' for col in columns})
    desc2 = desc2.rename(columns={col:f'{col}#2' for col in columns})
    desc = pd.concat([desc1, desc2], axis=1)
    return desc

def add_compound_descriptors(df, df_desc,function=False):
    columns = df_desc.drop(['Component','Smiles'], axis=1).columns.tolist()
    results = df.copy()
    for idx in results.index:
        smiles_1 = results.loc[idx,'Smiles#1']
        smiles_2 = results.loc[idx,'Smiles#2']
        if smiles_2 == 'C[N+](C)(C)CC(=O)[O-].O':
            smiles_2 = 'C[N+](C)(C)CC(=O)[O-]' 
        desc_1 = df_desc[df_desc['Smiles']==smiles_1].drop(['Component','Smiles'],axis=1).reset_index(drop=True)
        desc_2 = df_desc[df_desc['Smiles']==smiles_2].drop(['Component','Smiles'],axis=1).reset_index(drop=True)
        descs = add_separate(columns,desc_1,desc_2)
        x1 = results.loc[idx, 'X#1']
        if not function:    
            for col in columns:
                results.loc[idx, [f'{col}#1',f'{col}#2']] = descs.loc[0,[f'{col}#1',f'{col}#2']] 
        else: 
            for col in columns:
                name, value = ratio(col,descs,x1)
                results.loc[idx, f'{col}#{name}'] = value.iloc[0]
    return results

def add_mixture_descriptors(df, df_desc):
    columns = []
    for col in df_desc.drop(['Component#1','Smiles#1','Component#2','Smiles#2'], axis=1).columns:
        col = col.split('#')[0]
        if col not in columns:
            columns.append(col)
    results = df.copy()
    for idx in results.index:
        smiles_1 = results.loc[idx,'Smiles#1']
        smiles_2 = results.loc[idx,'Smiles#2']
        descs = df_desc[(df_desc['Smiles#1']==smiles_1) & (df_desc['Smiles#2']==smiles_2)].drop(['Component#1','Smiles#1','Component#2','Smiles#2'],axis=1).reset_index(drop=True)
        for col in columns:
            results.loc[idx, [f'{col}#1',f'{col}#2']] = descs.loc[0,[f'{col}#1',f'{col}#2']] 
    return results
  
#%% Class descriptors
class descriptors:
    def __init__(self, paths=paths):
        self.InfD = pd.read_csv(paths['InfDilut'])[['Component#1','Smiles#1','Component#2','Smiles#2','ln_gamma_InfD#1','ln_gamma_InfD#2']]
        self.rdkit = pd.read_csv(paths['rdkit'])
        self.mom = pd.read_csv(paths['mom'])[['Component','Smiles','MolWeight','WAPS','WANS','COSMO_charge','charge_p','charge_n','area_p','area_n','Area','Volume']]
        self.profile = pd.read_csv(paths['profile'])
        self.thermo = pd.read_csv(paths['thermo']).drop(['inchi','Type'], axis=1)
        
    def add_rdkit(self, df):
        return add_compound_descriptors(df, self.rdkit)
    
    def add_mom(self, df, flag=False):
        if not flag:
            self.mom = self.mom.drop(['charge_p','charge_n','area_p','area_n','Area','Volume'], axis=1)
        return add_compound_descriptors(df, self.mom)
    
    def add_profile(self, df, function=True):
        return add_compound_descriptors(df, self.profile, function=function)
    
    def add_thermo(self, df):
        return add_compound_descriptors(df, self.thermo)
    
    def add_InfD(self, df):
        return add_mixture_descriptors(df, self.InfD)
    
    def add_several(self, df, thermo=True, rdkit=True, mom=True, InfDilut=True, profile=True):
        df_results = df.copy()
        if thermo:
            df_results = self.add_thermo(df_results)
        if rdkit:
            df_results = self.add_rdkit(df_results)
        if mom:
            df_results = self.add_mom(df_results)
        if InfDilut:
            df_results = self.add_InfD(df_results)
        if profile:
            df_results = self.add_profile(df_results)
        return df_results
    
    def add_selected_features(self, df):
        features = ['T#1','T#2','lny','MolWeight#2','SpherocityIndex','RadiusOfGyration',
                    'NPR1#2','NPR2#2','InertialShapeFactor#2','ChargeIndex','PolarityIndex',
                    'SymmetricIndex_HB','SymmetricIndex_MF']
        features_init = df.columns.tolist()
        df_results = df.copy()
        df_results['X#2'] = 1-df_results['X#1']
        df_results = add_compound_descriptors(df_results, self.thermo[['Component','Smiles','T']])
        df_results = add_mixture_descriptors(df_results, self.InfD)
        df_results['lny'] = df_results['X#1']*df_results['ln_gamma_InfD#1']+df_results['X#2']*df_results['ln_gamma_InfD#2']
        df_results = add_compound_descriptors(df_results, self.mom[['Component','Smiles','MolWeight','charge_p','charge_n','area_p','area_n']])
        df_results['WAPS'] = (df_results['X#1']*df_results['charge_p#1']+df_results['X#2']*df_results['charge_p#2'])/(df_results['X#1']*df_results['area_p#1']+df_results['X#2']*df_results['area_p#2'])
        df_results['WANS'] = (df_results['X#1']*df_results['charge_n#1']+df_results['X#2']*df_results['charge_n#2'])/(df_results['X#1']*df_results['area_n#1']+df_results['X#2']*df_results['area_n#2'])
        df_results['ChargeIndex'] = df_results['WAPS']/df_results['WANS']
        df_results = add_compound_descriptors(df_results, self.rdkit[['Component','Smiles','RadiusOfGyration','SpherocityIndex','NPR1','NPR2','InertialShapeFactor']])
        df_results['RadiusOfGyration'] = np.sqrt((df_results['MolWeight#1']*(df_results['RadiusOfGyration#1'])**2+df_results['MolWeight#2']*(df_results['RadiusOfGyration#2'])**2)/(df_results['X#1']*df_results['MolWeight#1']+df_results['X#2']*df_results['MolWeight#2']))
        df_results['SpherocityIndex'] = (df_results['SpherocityIndex#1']**df_results['X#1'])*(df_results['SpherocityIndex#2']**df_results['X#2'])
        df_results = add_compound_descriptors(df_results, self.profile.drop(['s_profile_1','s_profile_7'],axis=1), function=False)
        df_results['SymmetricIndex_MF'] = (df_results['X#1']*df_results['s_profile_3#1']+df_results['X#2']*df_results['s_profile_3#2'])/(df_results['X#1']*df_results['s_profile_5#1']+df_results['X#2']*df_results['s_profile_5#2'])
        df_results['SymmetricIndex_HB'] = (df_results['X#1']*df_results['s_profile_2#1']+df_results['X#2']*df_results['s_profile_2#2'])/(df_results['X#1']*df_results['s_profile_6#1']+df_results['X#2']*df_results['s_profile_6#2'])
        df_results['PolarityIndex'] = df_results['X#1']*df_results['s_profile_4#1']/(df_results['X#2']*df_results['s_profile_4#2'])
        return df_results[features_init+features]
            



            
                
            
