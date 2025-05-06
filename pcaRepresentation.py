"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "star_classification.csv"  # Change to your actual file path
df = pd.read_csv(file_path)

# convert class labels to numeric for PCA visualization
df["class"] = df["class"].map({'GALAXY': 0, 'QSO': 1, 'STAR': 2})

# Select only numeric features while excluding target variable and filenames
feature_columns = [col for col in df.columns if col not in ["obj_ID", "spec_obj_ID", "class"]]

# standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_columns])

# apply PCA (reduce to 2 components for visualization)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for visualization
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["Stellar Object"] = df["class"].map({0: "GALAXY", 1: "QUASAR", 2: "STAR"})

# Scatter plot of PCA components
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_pca["PC1"], y=df_pca["PC2"], hue=df_pca["Stellar Object"], alpha=0.7, palette=["blue", "red"])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection of Features (2D)")
plt.legend(title="Stellar Object")
plt.savefig("PCA_representation.png")"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "star_classification.csv"  # Change to your actual file path
df = pd.read_csv(file_path)

# Convert class labels to numeric
df["class"] = df["class"].map({'GALAXY': 0, 'QSO': 1, 'STAR': 2})

# Select numeric feature columns (exclude identifiers and target)
feature_columns = [col for col in df.columns if col not in ["obj_ID", "spec_obj_ID", "class"]]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_columns])

# Apply PCA with 3 components
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Create DataFrame for visualization
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"])
df_pca["Stellar Object"] = df["class"].map({0: "GALAXY", 1: "QUASAR", 2: "STAR"})

# Color map for the classes
color_map = {"GALAXY": "blue", "QUASAR": "red", "STAR": "green"}
colors = df_pca["Stellar Object"].map(color_map)

# 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df_pca["PC1"], df_pca["PC2"], df_pca["PC3"],
           c=colors, alpha=0.7, s=50)

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
ax.set_title("3D PCA Projection of Stellar Object Features")

# Custom legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=color, markersize=8)
                   for label, color in color_map.items()]
ax.legend(handles=legend_elements, title="Stellar Object")

# Save figure
plt.tight_layout()
plt.savefig("PCA_representation_3D.png")
