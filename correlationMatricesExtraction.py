import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("star_classification.csv")


columns_of_interest = ['redshift', 'u', 'g', 'r', 'class']
df_filtered = df[columns_of_interest].copy()

# Encode 'class' as a numeric variable if it is categorical
dx_mapping = {'GALAXY': 0, 'QSO': 1, 'STAR': 2}
df_filtered['class'] = df_filtered['class'].map(dx_mapping)

# compute the correlation matrix
correlation_matrix = df_filtered.corr()

# Set up the matplotlib figure
plt.figure(figsize=(8, 6))

# Draw the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

# Title and display
plt.title("Correlation Matrix Heatmap")
plt.savefig("correlation_matrix.png")