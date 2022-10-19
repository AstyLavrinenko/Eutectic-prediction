''' Removing the outlets and statistic '''

import pandas as pd
import numpy as np
import seaborn as sns
from statannot import add_stat_annotation
import matplotlib.pyplot as plt

df = pd.read_csv('../data/DES_init_update.csv')
df.info()

Q1, Q3 = np.percentile(df['T_EP'], [25,75])
IQR = Q3 - Q1
outliers = np.where((df['T_EP'] > (Q3+1.5*IQR)) | (df['T_EP'] <= (Q1-1.5*IQR)))
df_out = df.iloc[outliers[0]]
#df.drop(outliers[0], inplace = True)

sns.boxplot(df['T_EP'])
ax = sns.boxplot(x='Type', y='T_EP', data=df, hue='PD')
ax, test_results = add_stat_annotation(ax, x='Type', y='T_EP', data=df, hue='PD',
                                   box_pairs=[(('Type III','No'), ('Type III','Yes')), (('Type V','No'), ('Type V','Yes')), (('IL mixture','Yes'), ('Type III','Yes')), (('IL mixture','Yes'), ('Type V','Yes')), (('Type III','Yes'), ('Type V','Yes')), (('Type III','No'), ('Type V','No'))],
                                   test='Mann-Whitney', text_format='star', loc='outside', verbose=2)
sns.boxplot(df['X#1'])
ax = sns.boxplot(x='Type', y='X#1', data=df, hue='PD')
ax, test_results = add_stat_annotation(ax, x='Type', y='X#1', data=df, hue='PD',
                                   box_pairs=[(('Type III','No'), ('Type III','Yes')), (('Type V','No'), ('Type V','Yes')), (('IL mixture','Yes'), ('Type III','Yes')), (('IL mixture','Yes'), ('Type V','Yes')), (('Type III','Yes'), ('Type V','Yes')), (('Type III','No'), ('Type V','No'))],
                                   test='Mann-Whitney', text_format='star', loc='outside', verbose=2)

#%%
compounds = pd.read_csv('../descriptors/compounds/measured/thermochem.csv')
data = compounds.groupby(by=['Type'])['Component'].count()
labels = [
    'IL',                          
    'alcohol',                     
    'amide',                     
    'amino acid',                 
    'aromatic carboxylic acid',   
    'monocarboxylic acid',   
    'other',         
    'phenol',         
    'polycarboxylic acid',
    'polyol', 
    'saccharide']

fig, ax = plt.subplots(figsize=(8,6),subplot_kw=dict(aspect="equal"))
wedges, texts = ax.pie(data)
kw = dict(arrowprops=dict(arrowstyle="-"),
          zorder=0, va="center")
perc = [str(round(e / s * 100., 1)) + '%' for s in (sum(data),) for e in data]
for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(labels[i] + ' \n' + perc[i], xy=(x, y), xytext=(1.2*np.sign(x), 1.5*y),
                horizontalalignment=horizontalalignment, **kw)
plt.savefig('../results/compounds.eps',dpi=1200)
plt.show()

#%%
df_unique = df.drop_duplicates(subset = ['Smiles#1', 'Smiles#2'])
fig, ax = plt.subplots(figsize=(8,6))
ax=sns.histplot(data=df_unique,x='Type',hue='PD',color=sns.color_palette('pastel'),multiple='dodge',shrink=0.8,stat='percent')
plt.savefig('../results/type.eps',dpi=1200)
plt.show()

