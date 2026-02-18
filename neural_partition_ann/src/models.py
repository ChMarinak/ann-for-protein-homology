# models.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class PartitionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLPClassifier(nn.Module):
    def __init__(self, d_in, n_out, hidden=64, layers=3):
        super().__init__()
        layers_list = []
        in_features = d_in
        for i in range(layers-1):
            layers_list.append(nn.Linear(in_features, hidden))
            layers_list.append(nn.ReLU())
            in_features = hidden
        layers_list.append(nn.Linear(in_features, n_out))  # output layer
        self.net = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.net(x)


def train_classifier(X, labels, m, epochs=10, batch_size=32, lr=1e-3, hidden=64, layers=3, device=None):
    """
    Train an MLP on points X ∈ R^d with KaHIP labels in [0, m-1].
    Returns trained model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = X.shape[1]
    dataset = PartitionDataset(X, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLPClassifier(d_in=d, n_out=m, hidden=hidden, layers=layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            running_loss += loss.item()

        print(f"[Epoch {epoch+1}/{epochs}] loss = {running_loss/len(loader):.4f}")

    return model
