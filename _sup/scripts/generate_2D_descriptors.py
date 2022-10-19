''' Generate 2D descriptors by RDKit '''

from get_mols import compounds
import rdkit_descriptors
import pandas as pd

df_init = pd.read_csv('../descriptors/compounds/measured/thermochem.csv')
components = compounds(df_init['Smiles'], df_init['inchi'])

list_2D=['NumAliphaticCarbocycles','NumAliphaticHeterocycles',
         'NumAromaticCarbocycles','NumAromaticHeterocycles','NumHAcceptors','NumHDonors',
         'NumHeteroatoms','MaxPartialCharge','MinPartialCharge','BalabanJ',
         'fr_NH0','fr_NH1','fr_NH2','fr_Ndealkylation1','fr_amide','fr_halogen','fr_phenol','fr_quatN']

desc_2D = rdkit_descriptors.compute_2D_desc(components.mols, list_2D)
desc_2D = pd.concat([df_init[['Component','Smiles']], desc_2D], axis=1)
desc_3D = rdkit_descriptors.compute_3D_desc(components.mols)
pd.concat([desc_2D, desc_3D], axis=1).to_csv('../descriptors/compounds/calculated/rdkit_descriptors.csv', index=False)
