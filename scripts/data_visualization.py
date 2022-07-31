''' Removing the outlets and statistic '''

import pandas as pd
import numpy as np
import seaborn as sns
from statannot import add_stat_annotation

#%%
df = pd.read_csv('../data/DES_init_update.csv')
df.info()
Q1 = np.percentile(df['T_EP'], 25, interpolation = 'midpoint')
Q3 = np.percentile(df['T_EP'], 75, interpolation = 'midpoint')
IQR = Q3 - Q1
upper = np.where(df['T_EP'] >= (Q3+1.5*IQR))
lower = np.where(df['T_EP'] <= (Q1-1.5*IQR))
outliers = pd.concat([df.iloc[upper[0]], df.iloc[lower[0]]])
#df.drop(upper[0], inplace = True)
#df.drop(lower[0], inplace = True)
#%%
sns.boxplot(df['T_EP'])
ax = sns.boxplot(x='Type', y='T_EP', data=df, hue='Phase_diagram')
ax, test_results = add_stat_annotation(ax, x='Type', y='T_EP', data=df, hue='Phase_diagram',
                                   box_pairs=[(('III','No'), ('III','Yes')), (('V','No'), ('V','Yes')), (('I','Yes'), ('III','Yes')), (('I','Yes'), ('V','Yes')), (('III','Yes'), ('V','Yes')), (('III','No'), ('V','No'))],
                                   test='Mann-Whitney', text_format='star', loc='outside', verbose=2)
sns.boxplot(df['X_1'])
ax = sns.boxplot(x='Type', y='X_1', data=df, hue='Phase_diagram')
ax, test_results = add_stat_annotation(ax, x='Type', y='X_1', data=df, hue='Phase_diagram',
                                   box_pairs=[(('III','No'), ('III','Yes')), (('V','No'), ('V','Yes')), (('I','Yes'), ('III','Yes')), (('I','Yes'), ('V','Yes')), (('III','Yes'), ('V','Yes')), (('III','No'), ('V','No'))],
                                   test='Mann-Whitney', text_format='star', loc='outside', verbose=2)

bins=np.array([min(df['T_EP']), 300, 400, 500, max(df['T_EP'])])
group_names=['under 300', '300-400', '400-500', 'over 500']
df['binning'] = pd.cut(df['T_EP'], bins, labels=group_names, include_lowest=True)
sns.boxplot(x='binning', y='T_EP', data=df, hue='Type')
df.groupby(by='binning').count()
df = pd.get_dummies(df,columns=['Type'])
