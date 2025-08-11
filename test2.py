import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# --------------------
# CONFIG
# --------------------
SEQ_LENGTH = 24 * 30   # 1 month (hours)
PRED_LENGTH = 24 * 7   # 1 week (hours)
BATCH_SIZE = 32
EPOCHS = 20000000
LR = 1e-4
HIDDEN_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------
# DATA PREP
# --------------------
class ConsumptionDataset(Dataset):
    def __init__(self, df, seq_length, pred_length):
        self.seq_length = seq_length
        self.pred_length = pred_length

        # Feature engineering: Convert datetime to cyclical features
        df["hour"] = df["DateTime"].dt.hour
        df["dayofweek"] = df["DateTime"].dt.dayofweek
        df["month"] = df["DateTime"].dt.month

        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Features: time encodings + consumption
        features = ["hour_cos", "dow_cos", "month_cos", "Consumption"]
        self.data = df[features].values.astype(np.float32)

    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]            # [seq_length, features]
        y = self.data[idx + self.seq_length : idx + self.seq_length + self.pred_length, -1]  # consumption only
        return torch.tensor(x), torch.tensor(y)

# --------------------
# MODEL
# --------------------
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, pred_length):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_length)

    def forward(self, x):
        out, _ = self.rnn(x)           # [B, seq_length, hidden_size]
        out = out[:, -1, :]            # last time step
        out = self.fc(out)             # [B, pred_length]
        return out

# --------------------
# TRAIN FUNCTION
# --------------------
def train_model(model, train_loader, val_loader, epochs, lr, device, scaler_y):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    record = 999999999999
    early_stop = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            print(xb.shape)
            xb, yb = xb.to(device), yb.to(device)
            xb = scaler_y.fit_transform(xb.reshape(-1, 1))
            print(xb.shape)
            yb = scaler_y.transform(yb.reshape(-1, 1))
            exit()
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_loss += criterion(preds, yb).item()
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss/len(train_loader):.4f}  Val Loss: {val_loss/len(val_loader):.4f}")
        if val_loss<record:
            record = val_loss
            early_stop = 0
            torch.save(model.state_dict(), "simple_rnn_consumption.pth")
            print("Model saved as simple_rnn_consumption.pth")
        else:
            early_stop +=1
            if early_stop == 10:
                print("stop training")
                exit()
        

    return model

# --------------------
# MAIN SCRIPT
# --------------------
if __name__ == "__main__":
    # Example: load dataset
    # Your CSV must have 'DateTime' and 'Consumption'
    df = pd.read_csv("./electricityConsumptionAndProductioction.csv", parse_dates=["DateTime"])
    df = df.sort_values("DateTime").reset_index(drop=True)

    dataset = ConsumptionDataset(df, SEQ_LENGTH, PRED_LENGTH)

    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = SimpleRNN(input_size=4, hidden_size=HIDDEN_SIZE, pred_length=PRED_LENGTH).to(DEVICE)

    model = train_model(model, train_loader, val_loader, EPOCHS, LR, DEVICE, scaler_y = StandardScaler())

    # torch.save(model.state_dict(), "simple_rnn_consumption.pth")
    # print("Model saved as simple_rnn_consumption.pth")
