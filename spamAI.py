import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# 1. Load and clean data
df = pd.read_csv('data/spam/spam.csv', encoding='latin-1')
df = df.iloc[:, :2]
df.columns = ['label', 'text']
df = df.dropna(subset=['label', 'text'])
df['label'] = df['label'].map({'ham': 0, 'spam': 1}).astype(int)

# 2. Vectorize text (TF-IDF, fixed-size dense features)
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
X = vectorizer.fit_transform(df['text']).toarray().astype('float32')
y = df['label'].values.astype('float32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Dataset
class SpamDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y).unsqueeze(1)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_loader = DataLoader(SpamDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader  = DataLoader(SpamDataset(X_test, y_test), batch_size=64)

# 4. Model
model = nn.Sequential(
    nn.Linear(X.shape[1], 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),
)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 5. Train for 5 epochs
for epoch in range(5):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)

    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = (model(xb) > 0.5).float()
            correct += (preds == yb).sum().item()
    acc = correct / len(y_test)
    print(f"Epoch {epoch+1}: loss={total_loss/len(y_train):.4f}, test_acc={acc:.4f}")

# 6. Export to current working directory
cwd = os.getcwd()
model_path = os.path.join(cwd, 'spam_model.pth')
vectorizer_path = os.path.join(cwd, 'tfidf_vectorizer.pkl')

torch.save(model.state_dict(), model_path)
joblib.dump(vectorizer, vectorizer_path)

print(f"Saved model      -> {model_path}")
print(f"Saved vectorizer -> {vectorizer_path}")
