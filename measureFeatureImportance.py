import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

X = pd.read_csv("star_classification.csv")
X = X.drop('obj_ID', axis=1)
X = X.drop('spec_obj_ID', axis=1)
y = X["class"]
X = X.drop('class', axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

model = GradientBoostingClassifier()
model.fit(X_train, y_train)

importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
feature_importance_df.sort_values(by='Importance', ascending=False)

print(feature_importance_df.to_string())

plt.figure(figsize=(8, 4))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel("Feature Importance")
plt.title("Feature Importance from Random Forest")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png")