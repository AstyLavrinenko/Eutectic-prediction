''' Fingerprints of moleculs by fingerprints '''
import rdkit_descriptors
import pandas as pd
from rdkit import Chem
from sklearn.manifold import TSNE

df_init = pd.read_csv('../descriptors/compounds/measured/thermochem.csv')
mols=[Chem.MolFromSmiles(smi) for smi in df_init['Smiles']]
df_fp = df_init[['Component', 'Smiles']]
fingerprints = rdkit_descriptors.compute_fp(mols,2,8192,bitstring=True)
n_components = 10
tsne = TSNE(method='exact',n_components = n_components,random_state = 42,n_jobs = -1)
x_dimensions = tsne.fit_transform(fingerprints)
tsnedf_res = pd.DataFrame(data = x_dimensions,columns = [f'fp{i}' for i in range(1,n_components+1)])
df_fp = pd.concat([df_fp, tsnedf_res], axis=1)
df_fp.to_csv('../descriptors/compounds/calculated/fingerprints.csv', index=False)