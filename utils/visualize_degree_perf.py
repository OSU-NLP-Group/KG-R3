import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

degrees = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
hits1 = [0.20213, 0.20222, 0.27843, 0.2836, 0.26808, 0.29503, 0.32538, 0.29988, 0.37958, 0.4145, 0.37906, 0.51422, 0.49918, 0.47101]
hits10 = [0.25532, 0.40222, 0.45481, 0.50048, 0.4944, 0.52851, 0.57004, 0.54929, 0.60869, 0.65342, 0.61191, 0.65542, 0.55592, 0.62629]

df = pd.DataFrame(list(zip(degrees, hits1, hits10)), columns =['degree', 'Hits@1', 'Hits@10'])

plt.figure()
sns.lineplot(data=df, x='degree', y='Hits@1', label='Hits@1')
sns.lineplot(data=df, x='degree', y='Hits@10', label='Hits@10')
plt.legend()
plt.tight_layout()
plt.xlabel(r'log$_2$(degree)')
plt.ylabel('Link Prediction performance')
plt.savefig('degree_analysis.png')