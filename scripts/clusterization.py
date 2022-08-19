''' Clusterization of moleculs by fingerprints '''
import rdkit_descriptors
import pandas as pd
from sklearn.cluster import Birch
from rdkit import Chem

df_init = pd.read_csv('../descriptors/compounds/measured/thermochem.csv')
mols=[Chem.MolFromSmiles(smi) for smi in df_init['Smiles']]
df_clustered = df_init[['Component', 'Smiles']]
fingerprints = rdkit_descriptors.compute_fp(mols,3,8192,bitstring=False)
clusterer = Birch(threshold=0.001, n_clusters=20)
df_clustered['cluster_birch'] = clusterer.fit_predict(fingerprints)
df_clustered.to_csv('../descriptors/compounds/calculated/clusterization.csv', index=False)