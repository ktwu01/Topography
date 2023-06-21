import matplotlib.pyplot as plt
import pandas as pd
import os

# Creates plot that compares global distributions of different datasets.

# check if folder exists
results_path = "./results/data_bias/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

# only landforms
# load and process data
df = pd.read_csv("data/data_bias.csv", sep='\t')


x = ["Kratzert", "Moeck", "Fan", "Global"]
#x = ['Global', 'Streamflow (Kratzert et al., 2023)', 'Groundwater recharge (Moeck et al., 2020)', 'Water table depth (Fan et al., 2013)']
y0 = 100*df.loc[0].values[1:]
y1 = 100*df.loc[1].values[1:]
y2 = 100*df.loc[2].values[1:]+1 # to compensate for rounding error
y3 = 100*df.loc[3].values[1:]+1
fig = plt.figure(figsize=(4, 2), constrained_layout=True)
width = 0.5
bar1 = plt.bar(x, y0, width=width, color='#f4efc8', label='Plains')
bar2 = plt.bar(x, y1, bottom=y0, width=width, color='#f6c8a3', label='Plateaus')
bar3 = plt.bar(x, y2, bottom=y0+y1, width=width, color='#74a747', label='Hills')
bar4 = plt.bar(x, y3, bottom=y0+y1+y2, width=width, color='#83786c', label='Mountains')
#plt.xlabel("Landform")
plt.ylabel('Percentage %')
plt.ylim([0,100])
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
#ax.spines['left'].set_visible(False)
#plt.legend()
plt.savefig(results_path + "data_bias_stacked"  + ".png", dpi=600, bbox_inches='tight')
plt.close()


# including aridity
# load and process data
df = pd.read_csv("data/data_bias_aridity.csv", sep='\t')

x = ["Kratzert", "Moeck", "Fan", "Global"]
#x = ['Global', 'Streamflow (Kratzert et al., 2023)', 'Groundwater recharge (Moeck et al., 2020)', 'Water table depth (Fan et al., 2013)']
y0 = 100*df.loc[0].values[1:]
y1 = 100*df.loc[1].values[1:]
y2 = 100*df.loc[2].values[1:]+1 # to compensate for rounding error
y3 = 100*df.loc[3].values[1:]+1
fig = plt.figure(figsize=(4, 2), constrained_layout=True)
width = 0.5
bar1 = plt.bar(x, y0, width=width, color='#01665e', label='Humid plains')
bar2 = plt.bar(x, y1, bottom=y0, width=width, color='#80cdc1', label='Arid plains')
bar3 = plt.bar(x, y2, bottom=y0+y1, width=width, color='#8c510a', label='Humid uplands')
bar4 = plt.bar(x, y3, bottom=y0+y1+y2, width=width, color='#dfc27d', label='Arid uplands')
#plt.xlabel("Landform")
plt.ylabel('Percentage %')
plt.ylim([0,100])
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
#ax.spines['left'].set_visible(False)
#plt.legend()
plt.savefig(results_path + "data_bias_stacked_aridity"  + ".png", dpi=600, bbox_inches='tight')
plt.close()
