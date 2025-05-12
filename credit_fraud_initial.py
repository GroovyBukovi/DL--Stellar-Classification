import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from collections import Counter

# === LOAD DATA ===
df = pd.read_csv("train.csv")
print(df.shape)
print(df['Cover_Type'].value_counts())

# === SELECT FEATURES ===
# features = ['u', 'g', 'r', 'i', 'z', 'redshift']
features = df.drop(columns=["Cover_Type"]).columns.tolist()

#features = ['V7', 'V10', 'V12', 'V14', 'V14', 'V16', 'V17', 'V20', 'V27']
target = 'Cover_Type'

X = df[features].values
y = df[target].values

# === ENCODE LABELS ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_map = dict(zip(le.classes_, le.transform(le.classes_)))
print("Label Encoding:", class_map)

# === STANDARDIZE FEATURES ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

'''
# === Undersample first, then apply SMOTE to achieve 10% fraud ===
under = RandomUnderSampler(sampling_strategy={0: 261000, 1: 492}, random_state=42)

# Step 2: SMOTE minority to 29000
smote = SMOTE(sampling_strategy={1: 29000}, random_state=42)


# Correct pipeline order
pipeline = Pipeline(steps=[('under', under), ('smote', smote)])


# Step 3: Apply the resampling
X_resampled, y_resampled = pipeline.fit_resample(X, y)

X_train_tensor = torch.tensor(X_resampled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_resampled, dtype=torch.long).to(device)

# Step 4: Check class distribution
print('Original dataset shape:', Counter(y))
print('Resampled dataset shape:', Counter(y_resampled))


# Save the new, resampled dataset

data_combined = np.hstack((X_resampled, y_resampled.reshape(-1, 1)))


num_features = X_resampled.shape[1]
feature_columns = [f'feature_{i}' for i in range(num_features)]
columns = feature_columns + ['label']

# Create the DataFrame
df_resampled = pd.DataFrame(data_combined, columns=columns)

# Save to CSV
df_resampled.to_csv('resampled_dataset.csv', index=False)
'''

# === CONVERT TO TENSORS ===
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# === DATALOADERS ===
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# === MLP MODEL ===
# class MLPClassifier(nn.Module):
#     def __init__(self, input_dim, hidden_dim=64, output_dim=3):
#         super(MLPClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.out = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.out(x)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=3, dropout_rate=0.2):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.out(x)

# === INITIALIZE MODEL ===
model = MLPClassifier(input_dim=X.shape[1], output_dim=len(class_map)).to(device)

# === LOSS & OPTIMIZER ===
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# === TRAIN FUNCTION ===
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# === EVALUATION FUNCTION (LOSS + ACCURACY) ===
def evaluate_loss_and_accuracy(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            output = model(X_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

# === TRAIN LOOP ===
num_epochs = 15
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    train_eval_loss, train_acc = evaluate_loss_and_accuracy(model, train_loader, criterion)
    test_loss, test_acc = evaluate_loss_and_accuracy(model, test_loader, criterion)

    train_losses.append(train_eval_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print(f"Epoch {epoch+1:2}/{num_epochs} - Train Loss: {train_eval_loss:.4f} - Test Loss: {test_loss:.4f} - Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}")

# === REPORT & CONFUSION MATRIX ===
def print_report(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            preds = model(X_batch)
            _, predicted = torch.max(preds, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    print(classification_report(all_labels, all_preds, target_names=[str(c) for c in le.classes_]))


print_report(model, test_loader)

def plot_confusion(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            preds = model(X_batch)
            _, predicted = torch.max(preds, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

plot_confusion(model, test_loader)
# === OPTIONAL: PLOT LOSS & ACCURACY CURVES ===
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Acc')
plt.plot(test_accuracies, label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.show()