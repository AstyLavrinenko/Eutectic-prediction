''' Merged correlated features '''

from sklearn.manifold import TSNE

features_merge = {'Geometry#2':['Area#2','MolWeight#2','Volume#2'],
                  'COSMO_energy':['E_COSMO#1','Free_energy_COSMO_InfD#1','Free_energy_COSMO_InfD#2',
                                  'H_int_InfD#1','H_int_InfD#2'],
                  'VdW_interaction':['E_vdw#2','H_vdW_InfD#1','H_vdW_InfD#2'],
                  'HB#1':['fr_Ar_N#1','fr_Ndealkylation1#1','NumAliphaticCarbocycles#1',
                          'NumAliphaticHeterocycles#1','NumAromaticCarbocycles#1',
                          'NumAromaticHeterocycles#1','NumHeteroatoms#1','H_HB_InfD#1'],
                  'HB#2':['fr_Ndealkylation1#2','fr_quatN#2','H_HB_InfD#2'],
                  'S_profile':['s_profile_2#1','s_profile_2#2','s_profile_3#1','s_profile_3#2',
                               's_profile_4#1','s_profile_4#2','s_profile_5#1','s_profile_5#2',
                               's_profile_6#1','s_profile_6#2','s_profile_7#1','s_profile_7#2',
                               's_profile_9#1','s_profile_9#2','s_profile_10#1','s_profile_10#2'],
                  'Charge#2':['NumValenceElectrons#2','WANS#2','WAPS#2']
                  }

features = ['Area#2','BCUT2D_MWLOW#1','E_COSMO#1','E_vdw#2','fr_Ar_N#1','fr_Ndealkylation1#1',
            'fr_Ndealkylation1#2','fr_quatN#2','Free_energy_COSMO_InfD#1','Free_energy_COSMO_InfD#2',
            'H_HB_InfD#1','H_HB_InfD#2','H_int_InfD#1','H_int_InfD#2','H_MF_InfD#1','H_vdW_InfD#1',
            'H_vdW_InfD#2','ln_gamma_InfD#1','ln_gamma_InfD#2','MolWeight#2',
            'NumAliphaticCarbocycles#1','NumAliphaticHeterocycles#1','NumAromaticCarbocycles#1',
            'NumAromaticHeterocycles#1','NumHeteroatoms#1','NumValenceElectrons#2',
            'partial_pressure_InfD#1','partial_pressure_InfD#2','RTlnx#1','RTlnx#2','s_profile_2#ratio',
            's_profile_3#ratio','s_profile_4#ratio','s_profile_5#ratio','s_profile_6#ratio',
            's_profile_7#ratio','s_profile_9#ratio','s_profile_10#ratio',
            'T#1','T#2','Volume#2','WANS#2','WAPS#1','WAPS#2']

def merge_features(df):
    df_merged = df[features]
    for key, value in features_merge.items():
        tsne = TSNE(n_components = 1,init='random',perplexity=15,learning_rate=100,random_state = 42,n_jobs = -1)
        x_dimensions = tsne.fit_transform(df_merged[value])
        df_merged[f'{key}'] = x_dimensions[:,0]
        #df_merged[f'{key}#Dim2'] = x_dimensions[:,1]
        df_merged = df_merged.drop(value, axis=1)
    return df_merged
#corr=pd.concat([df['T_EP'],df_merged],axis=1).corr()
