
# ================================
# STEP 1: INSTALL & IMPORT
# ================================

!pip install numpy pandas scikit-learn torch gpytorch matplotlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import gpytorch
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ================================
# STEP 2: LOAD & PREPROCESS DATA
# ================================

# Load dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# ================================
# STEP 3: BASELINE MLP MODEL
# ================================

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()

mlp = MLP()
optimizer = optim.Adam(mlp.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Train MLP
for epoch in range(100):
    optimizer.zero_grad()
    preds = mlp(X_train)
    loss = loss_fn(preds, y_train)
    loss.backward()
    optimizer.step()

# Evaluate MLP
mlp.eval()
with torch.no_grad():
    preds_mlp = mlp(X_test)

rmse_mlp = mean_squared_error(y_test, preds_mlp, squared=False)
mae_mlp = mean_absolute_error(y_test, preds_mlp)

print("MLP RMSE:", rmse_mlp)
print("MLP MAE:", mae_mlp)


# ================================
# STEP 4: DKL FEATURE EXTRACTOR
# ================================

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


# ================================
# STEP 5: DKL GP MODEL
# ================================

class DKLModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ================================
# STEP 6: TRAIN DKL MODEL
# ================================

likelihood = gpytorch.likelihoods.GaussianLikelihood()
feature_extractor = FeatureExtractor()

dkl_model = DKLModel(X_train, y_train, likelihood, feature_extractor)

dkl_model.train()
likelihood.train()

optimizer = torch.optim.Adam(dkl_model.parameters(), lr=0.01)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, dkl_model)

for i in range(50):
    optimizer.zero_grad()
    output = dkl_model(X_train)
    loss = -mll(output, y_train)
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f"Iteration {i} - Loss: {loss.item():.3f}")


# ================================
# STEP 7: DKL PREDICTION & UNCERTAINTY
# ================================

dkl_model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions = likelihood(dkl_model(X_test))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()


# ================================
# STEP 8: DKL EVALUATION
# ================================

rmse_dkl = mean_squared_error(y_test, mean, squared=False)
mae_dkl = mean_absolute_error(y_test, mean)

coverage = ((y_test >= lower) & (y_test <= upper)).float().mean()

print("DKL RMSE:", rmse_dkl)
print("DKL MAE:", mae_dkl)
print("95% CI Coverage:", coverage.item())


# ================================
# STEP 9: FINAL COMPARISON TABLE
# ================================

results = pd.DataFrame({
    "Model": ["MLP", "DKL"],
    "RMSE": [rmse_mlp, rmse_dkl],
    "MAE": [mae_mlp, mae_dkl],
    "Confidence Interval Coverage": ["N/A", f"{coverage.item()*100:.2f}%"]
})

results


