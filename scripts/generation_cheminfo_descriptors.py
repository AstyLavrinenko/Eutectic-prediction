''' Generate descriptors by RDKit '''
#%% Imports
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors, Descriptors3D, AllChem, rdMolDescriptors
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
#%% Functions
class RDKit_Desc:
    def __init__(self, smiles):
        self.mols = [Chem.MolFromSmiles(i) for i in smiles]
        self.smiles = smiles
        
    def compute_2D_desc(self, index):
        rdkit_2d_desc = []
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
        header = calc.GetDescriptorNames()
        for i in range(len(self.mols)):
            ds = calc.CalcDescriptors(self.mols[i])
            rdkit_2d_desc.append(ds)
        df = pd.DataFrame(rdkit_2d_desc,columns=[f'{head}#{index}' for head in header])
        return df
    
    def compute_3D_desc(self, index):
        rdkit_3d_desc = {smi: Chem.MolFromSmiles(smi) for smi in self.smiles}
        descList1 = {'PMI1' : Descriptors3D.PMI1, 'PMI2' : Descriptors3D.PMI2, 
                    'PMI3' : Descriptors3D.PMI3, 'NPR1' : Descriptors3D.NPR1, 
                    'NPR2' : Descriptors3D.NPR2, 'RadiusOfGyration' : Descriptors3D.RadiusOfGyration, 
                    'InertialShapeFactor' :  Descriptors3D.InertialShapeFactor, 
                    'Eccentricity' : Descriptors3D.Eccentricity, 
                    'Asphericity' : Descriptors3D.Asphericity, 'SpherocityIndex' : Descriptors3D.SpherocityIndex}
        descList2 = {'RDF': rdMolDescriptors.CalcRDF, 'MORSE': rdMolDescriptors.CalcMORSE, 
                     'WHIM': rdMolDescriptors.CalcWHIM}
        for smi, mol in rdkit_3d_desc.items():
            ps = AllChem.EmbedParameters()
            ps.embedFragmentsSeparately = False
            ps.useRandomCoords = True
            mol = Chem.AddHs(mol)
            flag = AllChem.EmbedMultipleConfs(mol, 10, ps)
            ds = [descList1[key](mol) for key in descList1.keys()]
            header = list(descList1.keys())
            ds_rdMol = {key: descList2[key](mol) for key in descList2.keys()}
            for key in ds_rdMol.keys():
                ds += [ds_rdMol[key][x] for x in range(len(ds_rdMol[key]))]
                header += [f'{key}_{x+1}' for x in range(len(ds_rdMol[key]))]
            print(len(header))
            rdkit_3d_desc[smi]=ds
        df = pd.DataFrame(columns=[f'{head}#{index}' for head in header], index=[idx for idx, smi in enumerate(self.smiles)])
        for idx, smi in enumerate(self.smiles):
            df.iloc[idx] = rdkit_3d_desc[smi]
        return df
#%% Main code
df_init = pd.read_csv('../data/DES_init_update.csv')
desc_2D_1 = RDKit_Desc(df_init['Smiles_1']).compute_2D_desc('1')
desc_2D_2 = RDKit_Desc(df_init['Smiles_2']).compute_2D_desc('2')
desc_2D = pd.concat([df_init[['Component_1','Smiles_1','Component_2','Smiles_2','T_EP','X_1']],desc_2D_1,desc_2D_2], axis=1)
desc_2D.to_csv('../descriptors/cheminfo_2D.csv', index=False)

desc_3D_1 = RDKit_Desc(df_init['Smiles_1']).compute_3D_desc('1')
desc_3D_2 = RDKit_Desc(df_init['Smiles_2']).compute_3D_desc('2')
desc_3D = pd.concat([df_init[['Component_1','Smiles_1','Component_2','Smiles_2','T_EP','X_1']],desc_3D_1,desc_3D_2], axis=1)
desc_3D.to_csv('../descriptors/cheminfo_3D.csv', index=False)
