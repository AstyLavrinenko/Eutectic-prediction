''' Removing the outlets and statistic '''

import pandas as pd
import numpy as np
import seaborn as sns
from statannot import add_stat_annotation

df = pd.read_csv('../data/DES_init_update.csv')
df.info()

Q1, Q3 = np.percentile(df['T_EP'], [25,75])
IQR = Q3 - Q1
outliers = np.where((df['T_EP'] > (Q3+1.5*IQR)) | (df['T_EP'] <= (Q1-1.5*IQR)))
df_out = df.iloc[outliers[0]]
#df.drop(outliers[0], inplace = True)

sns.boxplot(df['T_EP'])
ax = sns.boxplot(x='Type', y='T_EP', data=df, hue='Phase_diagram')
ax, test_results = add_stat_annotation(ax, x='Type', y='T_EP', data=df, hue='Phase_diagram',
                                   box_pairs=[(('III','No'), ('III','Yes')), (('V','No'), ('V','Yes')), (('I','Yes'), ('III','Yes')), (('I','Yes'), ('V','Yes')), (('III','Yes'), ('V','Yes')), (('III','No'), ('V','No'))],
                                   test='Mann-Whitney', text_format='star', loc='outside', verbose=2)
sns.boxplot(df['X#1'])
ax = sns.boxplot(x='Type', y='X#1', data=df, hue='Phase_diagram')
ax, test_results = add_stat_annotation(ax, x='Type', y='X#1', data=df, hue='Phase_diagram',
                                   box_pairs=[(('III','No'), ('III','Yes')), (('V','No'), ('V','Yes')), (('I','Yes'), ('III','Yes')), (('I','Yes'), ('V','Yes')), (('III','Yes'), ('V','Yes')), (('III','No'), ('V','No'))],
                                   test='Mann-Whitney', text_format='star', loc='outside', verbose=2)

