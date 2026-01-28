# ============================================================
# Advanced Time Series Forecasting with Deep Learning & Attention
# ============================================================

# -----------------------------
# 1. IMPORTS & SETUP
# -----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from statsmodels.tsa.statespace.sarimax import SARIMAX

np.random.seed(42)
torch.manual_seed(42)


# -----------------------------
# 2. SYNTHETIC DATA GENERATION
# -----------------------------
def generate_time_series(n_samples=2500):
    t = np.arange(n_samples)

    feature_1 = 0.05 * t + np.sin(2 * np.pi * t / 50)
    feature_2 = np.cos(2 * np.pi * t / 30)
    feature_3 = np.sin(2 * np.pi * t / 100)
    feature_4 = np.random.normal(0, 0.3, n_samples)
    feature_5 = 0.001 * (t ** 1.5)

    noise = np.random.normal(0, 0.2, n_samples)

    target = (
        0.4 * feature_1 +
        0.3 * feature_2 +
        0.2 * feature_3 +
        0.1 * feature_5 +
        noise
    )

    df = pd.DataFrame({
        "feature_1": feature_1,
        "feature_2": feature_2,
        "feature_3": feature_3,
        "feature_4": feature_4,
        "feature_5": feature_5,
        "target": target
    })
    return df


data = generate_time_series()

# -----------------------------
# 3. DATA VISUALIZATION
# -----------------------------
plt.figure(figsize=(12,4))
plt.plot(data["target"])
plt.title("Synthetic Non-Stationary Target Series")
plt.show()


# -----------------------------
# 4. TRAIN–TEST SPLIT
# -----------------------------
train_size = int(len(data) * 0.8)
train = data.iloc[:train_size]
test = data.iloc[train_size:]


# -----------------------------
# 5. BASELINE MODEL (SARIMAX)
# -----------------------------
sarimax_model = SARIMAX(
    train["target"],
    order=(1,1,1),
    seasonal_order=(1,1,1,12)
)

sarimax_fit = sarimax_model.fit(disp=False)
baseline_predictions = sarimax_fit.forecast(len(test))


# -----------------------------
# 6. EVALUATION FUNCTION
# -----------------------------
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape


baseline_mae, baseline_rmse, baseline_mape = evaluate(
    test["target"], baseline_predictions
)


# -----------------------------
# 7. DEEP LEARNING DATA PREP
# -----------------------------
SEQ_LEN = 20

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.X = data.iloc[:, :-1].values
        self.y = data["target"].values
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx:idx+self.seq_len], dtype=torch.float32),
            torch.tensor(self.y[idx+self.seq_len], dtype=torch.float32)
        )

train_ds = TimeSeriesDataset(train, SEQ_LEN)
test_ds = TimeSeriesDataset(test, SEQ_LEN)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)


# -----------------------------
# 8. ATTENTION-BASED LSTM MODEL
# -----------------------------
class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        output = self.fc(context)
        return output.squeeze(), attn_weights


model = AttentionLSTM(input_dim=5, hidden_dim=64)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# -----------------------------
# 9. TRAIN ADVANCED MODEL
# -----------------------------
EPOCHS = 15

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds, _ = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")


# -----------------------------
# 10. ADVANCED MODEL EVALUATION
# -----------------------------
model.eval()
preds = []
actuals = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        output, _ = model(X_batch)
        preds.extend(output.numpy())
        actuals.extend(y_batch.numpy())

advanced_mae, advanced_rmse, advanced_mape = evaluate(
    np.array(actuals),
    np.array(preds)
)


# -----------------------------
# 11. PERFORMANCE COMPARISON
# -----------------------------
comparison = pd.DataFrame({
    "Model": ["SARIMAX", "Attention LSTM"],
    "MAE": [baseline_mae, advanced_mae],
    "RMSE": [baseline_rmse, advanced_rmse],
    "MAPE": [baseline_mape, advanced_mape]
})

print(comparison)


# -----------------------------
# 12. ATTENTION WEIGHT ANALYSIS
# -----------------------------
sample_X, _ = test_ds[0]
sample_X = sample_X.unsqueeze(0)

_, attention_weights = model(sample_X)

plt.figure(figsize=(8,4))
plt.plot(attention_weights.squeeze().numpy())
plt.title("Attention Weights Across Time Steps")
plt.xlabel("Time Step")
plt.ylabel("Importance")
plt.show()





