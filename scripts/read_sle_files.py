''' Read SLE data '''

import os, re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from rdkit import Chem

paths = {'CA17':'../data/SLE_COSMO/SLE_CA_17',
         'CA19':'../data/SLE_COSMO/SLE_CA_19',
         'C+A17':'../data/SLE_COSMO/SLE_C+A_17',
         'C+A19':'../data/SLE_COSMO/SLE_C+A_19'}

def get_inchi_separate(smiles, inchi):
    for idx in smiles.index:
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles[idx]))
        frags = Chem.GetMolFrags(mol, asMols = True)
        if len(frags) == 1:
            continue
        elif len(frags) == 2:
            inchi[idx] = f'{Chem.inchi.MolToInchiKey(frags[0])}_{Chem.inchi.MolToInchiKey(frags[1])}'
        else:
            return(idx, 'Number of fragments should be 1 or 2')
    return inchi

def read_SLE_tab(path, path_2):
    if os.path.exists(f'{path}.tab'):
        f = open(f'{path}.tab', 'r')
        conc = re.findall(r'SLE for compound \w+: x\(\w+\) = (\d.\d+E?-?\d+)', f.read(), re.DOTALL)
        f = open(f'{path}.tab', 'r')
        temp = re.findall(r'Settings  job \d+ : T\= (\d+.\d\d)', f.read(), re.DOTALL)
        conc_1 = []
        conc_2 = []
        for i in range(len(conc)):
            if i % 2 == 0:
                conc_1.append(conc[i])
            else:
                conc_2.append(1 - float(conc[i]))
        for i in range(len(temp)):
            temp[i] = float(temp[i])
            conc_2[i] = float(conc_2[i])
            conc_1[i] = float(conc_1[i])
    elif os.path.exists(f'{path_2}.tab'):
        f = open(f'{path_2}.tab', 'r')
        conc = re.findall(r'SLE for compound \w+: x\(\w+\) = (\d.\d+E?-?\d+)', f.read(), re.DOTALL)
        f = open(f'{path_2}.tab', 'r')
        temp = re.findall(r'Settings  job \d+ : T\= (\d+.\d\d)', f.read(), re.DOTALL)
        conc_1 = []
        conc_2 = []
        for i in range(len(conc)):
            if i % 2 == 0:
                conc_1.append(conc[i])
            else:
                conc_2.append(1 - float(conc[i]))
        for i in range(len(temp)):
            temp[i] = float(temp[i])
            conc_2[i] = float(conc_2[i])
            conc_1[i] = float(conc_1[i])
    else:
        temp = None
        conc_1 = None
        conc_2 = None
        print(f'{path}.tab does not exist')
    return temp, conc_1, conc_2

def find_EP(temp, conc_1, conc_2):
    first_line = LineString(np.column_stack((conc_2, temp)))
    second_line = LineString(np.column_stack((conc_1, temp)))
    intersection = first_line.intersection(second_line)
    if intersection.geom_type == 'MultiPoint':
        flag = 'Multiple intersection'
        x, y = LineString(intersection).xy
        a = []
        b = []
        for j in range(len(x)):
            a.append(round(x[j], 3))
            b.append(round(y[j], 3))
        T_calc = f'{b}'
        mark = len(b)
        X_calc = np.nan
    elif intersection.geom_type == 'Point':
        flag = 'Single intersection'
        x, y = intersection.xy
        X_calc = x[0]
        T_calc = y[0]
        mark = 1
    else:
        flag = 'Without intersection'
        T_calc = 'Zero intersection'
        mark = 0
        X_calc = np.nan
    return flag, T_calc, X_calc, mark

def plot_SLE(path, path_2, name_1, name, label, save=True):
    temp, conc_1, conc_2 = read_SLE_tab(path, path_2)
    if temp == None:
        T_calc=np.nan
        X_calc=np.nan
        mark=np.nan
        return T_calc, X_calc, mark
    else:
        flag, T_calc, X_calc, mark = find_EP(temp, conc_1, conc_2)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(conc_2, temp, '-or', conc_1, temp,'-ob')
        ax.set_xlabel(f"Mole fraction of {name_1}")
        ax.set_ylabel('Temperature, K')
        ax.set_yticks(np.arange(30, 680, 100))
        ax.set_xticks(np.arange(0, 1.02, 0.2))
        if flag == 'Single intersection':
            ax.set_title(f'{name} \n mole fraction = {round(X_calc, 3)}, T = {round(T_calc, 3)} K')
        else:
            ax.set_title(f'{name} \n {flag}')
        if save:
            fig.savefig(f'../results/SLE_plots/{label}/{name}.jpg')
        plt.show()
        return T_calc, X_calc, mark 

class DES:
    def __init__(self, df, flag):
        self.df = pd.DataFrame(index=df.index)
        self.flag = flag
        if self.flag not in ['CA17', 'CA19', 'C+A17', 'C+A19']:
            print('flag should be CA or C+A')
        else:
            names_1 = df.copy()['Component#1']
            names_2 = df.copy()['Component#2']
            inchi_1 = df.copy()['inchi#1']
            inchi_2 = df.copy()['inchi#2']
            if self.flag == 'C+A17' or self.flag == 'C+A19' :
                smiles_1 = df.copy()['Smiles#1']
                smiles_2 = df.copy()['Smiles#2']
                for smi in df.index:
                    if smiles_2[smi] == 'C[N+](C)(C)CC(=O)[O-].O':
                        smiles_2[smi] = 'C[N+](C)(C)CC(=O)[O-]'
                inchi_1 = get_inchi_separate(smiles_1, inchi_1)
                inchi_2 = get_inchi_separate(smiles_2, inchi_2)
            self.names = [f'{names_1[i]}#{names_2[i]}' for i in df.index]
            self.inchi = [f'{inchi_1[i]}_{inchi_2[i]}' for i in df.index]
    def plot(self):
        path_inp = paths[self.flag]
        path_inp_2 = paths[f'CA{self.flag.split("A")[1]}']
        T_SLE = []
        X_SLE = []
        mark_SLE = []
        for idx, name in enumerate(self.names):
            name_1 = name.split('#')[0]
            path = f'{path_inp}/{self.inchi[idx]}'
            path_2 = f'{path_inp_2}/{self.inchi[idx]}'
            T_calc, X_calc, mark = plot_SLE(path, path_2, name_1, name, self.flag, save=True)
            T_SLE.append(T_calc)
            X_SLE.append(X_calc)
            mark_SLE.append(mark)
        self.df[f'T_{self.flag}'] = T_SLE
        self.df[f'X_{self.flag}'] = X_SLE
        self.df[f'mark_{self.flag}'] = mark_SLE
        return self.df
                
            
