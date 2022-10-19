''' Descriptors by RDKit '''

import pandas as pd
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Descriptors3D, rdMolDescriptors

class compounds:
    def __init__(self, smiles, inchi):
        self.smiles = smiles            
        self.inchi = inchi
        self.mols = [Chem.AddHs(Chem.MolFromSmiles(i)) for i in self.smiles]
        for mol in self.mols:
            ps = AllChem.EmbedParameters()
            ps.embedFragmentsSeparately = False
            ps.useRandomCoords = True
            flag = AllChem.EmbedMultipleConfs(mol, 10, ps)

    def compute_2D_desc(self,list_descriptors_2D=[x[0] for x in Descriptors._descList]):
        rdkit_2d_desc = []
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(list_descriptors_2D)
        header = calc.GetDescriptorNames()
        for mol in self.mols:
            ds = calc.CalcDescriptors(mol)
            rdkit_2d_desc.append(ds)
        df = pd.DataFrame(rdkit_2d_desc,columns=header)
        return df
    
    def compute_3D_desc(self,list_descriptors_3D=['NPR1','NPR2','RadiusOfGyration','InertialShapeFactor',
                                               'SpherocityIndex']):
        rdkit_3d_desc = []
        descList1 = {'PMI1' : Descriptors3D.PMI1, 'PMI2' : Descriptors3D.PMI2, 
                 'PMI3' : Descriptors3D.PMI3, 'NPR1' : Descriptors3D.NPR1, 
                 'NPR2' : Descriptors3D.NPR2, 'RadiusOfGyration' : Descriptors3D.RadiusOfGyration, 
                 'InertialShapeFactor' :  Descriptors3D.InertialShapeFactor, 
                 'Eccentricity' : Descriptors3D.Eccentricity, 
                 'Asphericity' : Descriptors3D.Asphericity, 
                 'SpherocityIndex' : Descriptors3D.SpherocityIndex}
        descList2 = {'RDF': rdMolDescriptors.CalcRDF, 'MORSE': rdMolDescriptors.CalcMORSE, 
                 'WHIM': rdMolDescriptors.CalcWHIM}
        for mol in self.mols:
            header=[]
            ds=[]
            for key, func in descList1.items():
                if key in list_descriptors_3D:
                    header.append(key)
                    ds.append(func(mol))
                else:
                    continue
            for key, func in descList2.items():
                if key in list_descriptors_3D:
                    ds_rdMol=func(mol)
                    header+=[f'{key}_{x+1}' for x in range(len(ds_rdMol))]
                    ds+=ds_rdMol
            rdkit_3d_desc.append(ds)
        df = pd.DataFrame(rdkit_3d_desc,columns=header)
        return df

    