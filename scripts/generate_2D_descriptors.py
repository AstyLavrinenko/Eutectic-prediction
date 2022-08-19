''' Generate 2D descriptors by RDKit '''

from get_mols import compounds
import rdkit_descriptors
import pandas as pd

df_init = pd.read_csv('../descriptors/compounds/measured/thermochem.csv')
components = compounds(df_init['Smiles'], df_init['inchi'])

list_2D=['NumValenceElectrons','TPSA','NumAliphaticCarbocycles','NumAliphaticHeterocycles',
         'NumAromaticCarbocycles','NumAromaticHeterocycles','NumHAcceptors','NumHDonors',
         'NumHeteroatoms','NumRotatableBonds','MaxPartialCharge','MinPartialCharge','BalabanJ',
         'BCUT2D_MWHI','BCUT2D_MWLOW','BCUT2D_CHGHI','BCUT2D_CHGLO','BCUT2D_LOGPHI','BCUT2D_LOGPLOW',
         'BCUT2D_MRHI','BCUT2D_MRLOW', 'fr_Al_COO','fr_Al_OH','fr_Ar_COO','fr_Ar_N','fr_Ar_OH',
         'fr_C_O_noCOO','fr_NH0','fr_NH1','fr_NH2','fr_Ndealkylation1','fr_amide','fr_halogen',
         'fr_phenol','fr_quatN']

desc_2D = rdkit_descriptors.compute_2D_desc(components.mols, list_2D)
desc_2D = pd.concat([df_init[['Component','Smiles']], desc_2D], axis=1)
desc_2D.to_csv('../descriptors/compounds/calculated/2D_descriptors.csv', index=False)