####################### 3D REPRESENTATION ####################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Load Data ===
df = pd.read_csv("resampled_dataset.csv")

# === Extract Features and Labels ===
X = df.drop(columns=["feature_0", "label"])
y = df["label"]

# === Standardize Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === PCA Transformation ===
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# === Prepare DataFrame for Plotting ===
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])
pca_df['label'] = y.values

# === Plot PCA Result ===
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

colors = {0: 'green', 1: 'red'}
labels = {0: 'Non-Fraud', 1: 'Fraud'}

for cls in [0, 1]:
    subset = pca_df[pca_df['label'] == cls]
    ax.scatter(subset['PC1'], subset['PC2'], subset['PC3'],
               c=colors[cls], label=labels[cls], s=1, alpha=0.6)

ax.set_title('PCA (3 Components) - Credit Card Transactions')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.legend()

# === Save Plot ===
plt.tight_layout()
plt.savefig("pca_3d_resampled.png", dpi=300)
plt.show()

####################### 2D REPRESENTATION #####################################

"""import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# === Load Data ===
df = pd.read_csv("resampled_dataset.csv")

# === Extract Features and Labels ===
X = df.drop(columns=["feature_0", "label"])
y = df["label"]

# === Standardize Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === PCA Transformation (2D) ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# === Prepare DataFrame for Plotting ===
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['label'] = y.values

# === Plot 2D PCA Result ===
plt.figure(figsize=(10, 7))
colors = {0: 'green', 1: 'red'}
labels = {0: 'Non-Fraud', 1: 'Fraud'}

for cls in [0, 1]:
    subset = pca_df[pca_df['label'] == cls]
    plt.scatter(subset['PC1'], subset['PC2'],
                c=colors[cls], label=labels[cls], s=1, alpha=0.6)

plt.title('PCA (2 Components) - Credit Card Transactions')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.tight_layout()

# === Save Plot ===
plt.savefig("pca_2d_resampled.png", dpi=300)
plt.show()
"""