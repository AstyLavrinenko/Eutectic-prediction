''' Generate descriptors by RDKit '''
import pandas as pd
import numpy as np
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import AllChem, Descriptors, Descriptors3D, rdMolDescriptors
from rdkit import DataStructs
                   
def compute_2D_desc(mols, list_descriptors_2D=[x[0] for x in Descriptors._descList]):
    rdkit_2d_desc = []
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(list_descriptors_2D)
    header = calc.GetDescriptorNames()
    for mol in mols:
        ds = calc.CalcDescriptors(mol)
        rdkit_2d_desc.append(ds)
    df = pd.DataFrame(rdkit_2d_desc,columns=header)
    return df
    
def compute_3D_desc(mols, list_descriptors_3D=['PMI1','PMI2','PMI3','NPR1','NPR2', 
                                                       'RadiusOfGyration','InertialShapeFactor', 
                                                       'Eccentricity','Asphericity', 
                                                       'SpherocityIndex','RDF','MORSE','WHIM']):
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
    for mol in mols:
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

def compute_fp(mols,fp_radius,fp_length,bitstring=True):
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol,fp_radius,nBits=fp_length,useFeatures=True) for mol in mols]
    if bitstring:
        header=[f'fp{idx_fp}' for idx_fp in range(fp_length)]
        rdkit_fp=[np.array(fp) for fp in fps]
        df = pd.DataFrame(rdkit_fp,columns=header)
    else:
        dists = []
        for fp1 in fps:
            sims = [DataStructs.DiceSimilarity(fp1,fp2) for fp2 in fps]
            dists.append(sims)
        df = pd.DataFrame(dists)
    return df
    