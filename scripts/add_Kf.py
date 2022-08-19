''' add Kf '''
import pandas as pd
df = pd.read_csv('../descriptors/compounds/measured/thermochem.csv')
R = 8.31446261815324
for idx in df.index:
    df.loc[idx, 'Kf'] = R*df.loc[idx, 'T']**2/(1000*df.loc[idx, 'H'])
df.to_csv('../descriptors/compounds/measured/thermochem.csv', index=False)