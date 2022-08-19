''' Get mol files '''
import io, re
import pandas as pd
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import AllChem

class compounds:
    def __init__(self, smiles, inchi):
        self.smiles = smiles
        self.inchi = inchi
        self.mols = [Chem.AddHs(Chem.MolFromSmiles(i)) for i in self.smiles]
        for idx_mol, path in enumerate(self.inchi):
            with open(f'../data/logs/{path}.log', 'r') as inpf:
                text = inpf.read()
            segments = re.findall(r'[^\n]+Redundant internal[^\n]+\n(.+)\n\s+Recover', text, re.DOTALL)
            if len(segments) != 1:
                print(f'{path}: error')
                continue
            cols = ['atom_type','c', 'X','Y','Z']
            df = pd.read_csv(io.StringIO(segments[0]), names = cols, sep = r'\,', engine = 'python').drop(['c'], axis=1)
            ps = AllChem.EmbedParameters()
            ps.embedFragmentsSeparately = False
            ps.useRandomCoords = True
            flag = AllChem.EmbedMultipleConfs(self.mols[idx_mol], 10, ps)
            for i in range(self.mols[idx_mol].GetNumAtoms()):
                x,y,z = df.loc[i,['X','Y','Z']]
                self.mols[idx_mol].GetConformer().SetAtomPosition(i,Point3D(x,y,z))


