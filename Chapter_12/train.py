import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib


ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)


class IrisClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.network(x)


def main():
    iris = load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    model = IrisClassifier()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()

    for epoch in range(200):
        optimizer.zero_grad()

        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        logits = model(X_test_tensor)
        predictions = torch.argmax(logits, dim=1).numpy()

    accuracy = accuracy_score(y_test, predictions)

    torch.save(model.state_dict(), ARTIFACT_DIR / "iris_model.pt")
    joblib.dump(scaler, ARTIFACT_DIR / "scaler.joblib")

    metadata = {
        "model_version": "1.0.0",
        "accuracy": float(accuracy),
        "classes": iris.target_names.tolist(),
        "input_features": [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]
    }

    with open(ARTIFACT_DIR / "metadata.json", "w") as file:
        json.dump(metadata, file, indent=4)

    print(f"Model trained successfully. Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()