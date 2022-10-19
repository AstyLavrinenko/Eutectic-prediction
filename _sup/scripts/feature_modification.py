''' Feature modification '''
#%% Imports
import add_descriptors
import fit_evaluate

import pandas as pd
import numpy as np
#%%
df = pd.read_csv('../descriptors/mixture/main.csv')
df = add_descriptors.descriptors().add_several(df, mom=False, profile=False)
df = add_descriptors.descriptors().add_mom(df, flag=True)
df = add_descriptors.descriptors().add_profile(df, function=False)

#%%
df['X#2'] = 1-df['X#1']
df['RadiusOfGyration'] = np.sqrt((df['MolWeight#1']*(df['RadiusOfGyration#1'])**2+df['MolWeight#2']*(df['RadiusOfGyration#2'])**2)/(df['X#1']*df['MolWeight#1']+df['X#2']*df['MolWeight#2']))
df['SpherocityIndex'] = (df['SpherocityIndex#1']**df['X#1'])*(df['SpherocityIndex#2']**df['X#2'])
df['lny'] = df['X#1']*df['ln_gamma_InfD#1']+df['X#2']*df['ln_gamma_InfD#2']
df['WAPS'] = (df['X#1']*df['charge_p#1']+df['X#2']*df['charge_p#2'])/(df['X#1']*df['area_p#1']+df['X#2']*df['area_p#2'])
df['WANS'] = (df['X#1']*df['charge_n#1']+df['X#2']*df['charge_n#2'])/(df['X#1']*df['area_n#1']+df['X#2']*df['area_n#2'])
df['ChargeIndex'] = df['WAPS']/df['WANS']
df['SymmetricIndex_MF'] = (df['X#1']*df['s_profile_3#1']+df['X#2']*df['s_profile_3#2'])/(df['X#1']*df['s_profile_5#1']+df['X#2']*df['s_profile_5#2'])
df['SymmetricIndex_HB'] = (df['X#1']*df['s_profile_2#1']+df['X#2']*df['s_profile_2#2'])/(df['X#1']*df['s_profile_6#1']+df['X#2']*df['s_profile_6#2'])
df['PolarityIndex'] = df['X#1']*df['s_profile_4#1']/(df['X#2']*df['s_profile_4#2'])
features = ['X#1','T#1','T#2','lny','MolWeight#2','SpherocityIndex','RadiusOfGyration',
            'NPR1#2','NPR2#2','InertialShapeFactor#2','ChargeIndex','PolarityIndex',
            'SymmetricIndex_HB','SymmetricIndex_MF']

y = df['T_EP']
x = df[features]
cv = fit_evaluate.custom_cv(x, y, df.groups, 1, 0.2)
for train_idx, test_idx in cv:
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
pd.concat([y_train,x_train],axis=1).corr().to_csv('../results/corr_new.csv')


