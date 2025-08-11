# electricity_lstm_pytorch.py
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import datetime

# ---------------------------
# Configuration & Seeding
# ---------------------------
CONFIG = {
    'INPUT_WEEKS': 4,
    'OUTPUT_WEEKS': 1,
    'LSTM_UNITS': 64,
    'DROPOUT_RATE': 0.2,
    'BATCH_SIZE': 32,
    'MAX_EPOCHS': 100,
    'PATIENCE': 10,
    'MIN_DELTA': 0.001,
    'LEARNING_RATE': 1e-3,
    'MODEL_DIR': './models'
}
os.makedirs(CONFIG['MODEL_DIR'], exist_ok=True)

SEQUENCE_LENGTH = CONFIG['INPUT_WEEKS'] * 7 * 24
FORECAST_HORIZON = CONFIG['OUTPUT_WEEKS'] * 7 * 24

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------------------
# Device & Mixed Precision
# ---------------------------
def configure_torch():
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')
    if use_gpu:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU found, using CPU")
    return device

device = configure_torch()
USE_AMP = torch.cuda.is_available()  # automatic mixed precision if GPU available

# ---------------------------
# Data Loading / Preprocessing
# ---------------------------
def load_electricity_data(file_path):
    print(f"Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset shape: {df.shape}")
        print("Columns:", list(df.columns))
        # Try to infer datetime & consumption columns
        datetime_candidates = ['datetime', 'date', 'time', 'timestamp', 'dt']
        consumption_candidates = ['consumption', 'consume', 'demand', 'load', 'usage', 'kwh', 'mwh']

        datetime_col = None
        consumption_col = None

        for col in df.columns:
            if any(c in col.lower() for c in datetime_candidates):
                datetime_col = col
                break
        for col in df.columns:
            if any(c in col.lower() for c in consumption_candidates):
                consumption_col = col
                break

        if datetime_col is None:
            datetime_col = df.columns[0]
            print(f"Warning: using '{datetime_col}' as datetime column (not auto-detected)")
        if consumption_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            consumption_col = [c for c in numeric_cols if c != datetime_col][0]
            print(f"Warning: using '{consumption_col}' as consumption column (not auto-detected)")

        print(f"Using DateTime: {datetime_col}, consumption: {consumption_col}")

        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df_processed = pd.DataFrame({
            'DateTime': df[datetime_col],
            'consumption': df[consumption_col]
        }).sort_values('DateTime').reset_index(drop=True)

        initial_rows = len(df_processed)
        df_processed = df_processed.dropna()
        if len(df_processed) < initial_rows:
            print(f"Removed {initial_rows - len(df_processed)} rows with missing values")

        if df_processed['consumption'].min() < 0:
            print("Warning: negative consumption values found")

        # compute most common freq
        if len(df_processed) > 1:
            df_processed['time_diff'] = df_processed['DateTime'].diff()
            most_common_freq = df_processed['time_diff'].mode()[0]
            print("Most common time frequency:", most_common_freq)
            df_processed = df_processed.drop(columns=['time_diff'])
        else:
            print("Not enough rows to infer frequency")

        print("Processed dataset info:")
        print(f"  Shape: {df_processed.shape}")
        print(f"  Date range: {df_processed['DateTime'].min()} to {df_processed['DateTime'].max()}")
        print(f"  Consumption range: {df_processed['consumption'].min():.2f} to {df_processed['consumption'].max():.2f}")
        print(f"  Average consumption: {df_processed['consumption'].mean():.2f}")

        return df_processed

    except FileNotFoundError:
        print(f"Error: '{file_path}' not found.")
        return None
    except Exception as e:
        print("Error loading data:", e)
        return None

def prepare_sequences(data_array, seq_len):
    sequences = []
    targets = []
    for i in range(len(data_array) - seq_len):
        sequences.append(data_array[i:i+seq_len])
        targets.append(data_array[i+seq_len])
    return np.array(sequences), np.array(targets)

# ---------------------------
# PyTorch Dataset
# ---------------------------
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------------
# Model Definition
# ---------------------------
class SimpleRNNDepth(nn.Module):
    def __init__(self, in_channels=3, hidden_size=128):
        super(SimpleRNNDepth, self).__init__()

        # Feature extractor (lightweight CNN)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),  # H/2
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),           # H/4
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),          # H/8
            nn.ReLU(inplace=True)
        )

        # Collapse spatial dims for RNN
        self.flatten = nn.Flatten(start_dim=2)

        # Simple RNN over time
        self.rnn = nn.RNN(
            input_size=128 *  (32) * (64),  # depends on input size
            hidden_size=hidden_size,
            batch_first=True
        )

        # Decoder to predict depth
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1 * 256 * 512)  # final depth map size (adjust as needed)
        )

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        features = []

        for t in range(T):
            f = self.encoder(x[:, t])  # [B, 128, H/8, W/8]
            f = self.flatten(f)        # [B, feature_dim]
            features.append(f)

        features = torch.stack(features, dim=1)  # [B, T, feature_dim]

        rnn_out, _ = self.rnn(features)          # [B, T, hidden_size]

        depth_maps = []
        for t in range(T):
            d = self.decoder(rnn_out[:, t])      # [B, H*W]
            d = d.view(B, 1, 256, 512)           # reshape to depth map
            depth_maps.append(d)

        return torch.stack(depth_maps, dim=1)    # [B, T, 1, H, W]

# ---------------------------
# Training Utilities
# ---------------------------
def train_model(model, train_loader, val_loader, config, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])
    criterion = nn.SmoothL1Loss()  # Huber-like
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=max(1, config['PATIENCE']//2), verbose=True, min_lr=1e-6)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_path = os.path.join(config['MODEL_DIR'], 'best_electricity_complex_lstm_model.pt')

    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None
    history = {'train_loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}

    for epoch in range(1, config['MAX_EPOCHS'] + 1):
        model.train()
        train_losses = []
        train_maes = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            if USE_AMP:
                with torch.cuda.amp.autocast():
                    preds = model(xb)
                    loss = criterion(preds, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

            train_losses.append(loss.item())
            train_maes.append(mean_absolute_error(yb.detach().cpu().numpy(), preds.detach().cpu().numpy()))

        # validation
        model.eval()
        val_losses = []
        val_maes = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                if USE_AMP:
                    with torch.cuda.amp.autocast():
                        preds = model(xb)
                        loss = criterion(preds, yb)
                else:
                    preds = model(xb)
                    loss = criterion(preds, yb)
                val_losses.append(loss.item())
                val_maes.append(mean_absolute_error(yb.cpu().numpy(), preds.cpu().numpy()))

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_mae = np.mean(train_maes)
        val_mae = np.mean(val_maes)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['mae'].append(train_mae)
        history['val_mae'].append(val_mae)

        print(f"Epoch {epoch}/{config['MAX_EPOCHS']} - train_loss: {train_loss:.6f} val_loss: {val_loss:.6f} train_mae: {train_mae:.4f} val_mae: {val_mae:.4f}")

        scheduler.step(val_loss)

        # early stopping logic
        if val_loss + config['MIN_DELTA'] < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'epoch': epoch}, best_path)
            print(f"  Saved best model to {best_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config['PATIENCE']:
                print(f"Early stopping at epoch {epoch}. No improvement in val_loss for {epochs_no_improve} epochs.")
                break

    # load best model
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
    return model, history

# ---------------------------
# Forecast helper (autoregressive)
# ---------------------------
def forecast_future(model, last_sequence, steps, scaler, device):
    model.eval()
    preds = []
    seq = last_sequence.copy()
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(seq.reshape(1, -1, 1), dtype=torch.float32).to(device)
            if USE_AMP:
                with torch.cuda.amp.autocast():
                    p = model(x).cpu().numpy().ravel()[0]
            else:
                p = model(x).cpu().numpy().ravel()[0]
            preds.append(p)
            seq = np.append(seq[1:], p)
    # inverse transform
    preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds_inv

# ---------------------------
# Metrics
# ---------------------------
def calculate_metrics(actual, predicted, dataset_name):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    # mape safe: avoid division by zero
    actual_safe = np.where(actual == 0, 1e-6, actual)
    mape = np.mean(np.abs((actual - predicted) / actual_safe)) * 100
    r2 = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2 + 1e-9))
    print(f"\n{dataset_name} Metrics:")
    print(f"  MAE: {mae:.2f} kWh")
    print(f"  RMSE: {rmse:.2f} kWh")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²: {r2:.4f}")
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == '__main__':
    print("=== Advanced Electricity Consumption Forecasting (PyTorch) ===")
    recommendations = {
        'input_periods': {'1_week':168, '2_weeks':336, '1_month':720, '2_months':1440},
        'output_periods': {'1_day':24, '3_days':72, '1_week':168, '1_month':720}
    }
    print("Recommendations sample:", recommendations)

    # Load CSV (change path as needed)
    csv_path = './electricityConsumptionAndProductioction.csv'
    df = load_electricity_data(csv_path)
    if df is None:
        print("Failed to load data. Exiting.")
        exit()

    # Quick exploration (sampling)
    sample_size = min(1000, len(df))
    sample_idx = np.linspace(0, len(df)-1, sample_size, dtype=int)
    plt.figure(figsize=(12,4))
    plt.plot(df.iloc[sample_idx]['DateTime'], df.iloc[sample_idx]['consumption'], alpha=0.7)
    plt.title("Time Series Sample")
    plt.xlabel("Date")
    plt.ylabel("Consumption")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Robust-ish scaling
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_consumption = scaler.fit_transform(df['consumption'].values.reshape(-1, 1)).flatten()

    total_samples = len(scaled_consumption) - SEQUENCE_LENGTH
    if total_samples <= 0:
        raise ValueError("Not enough data for the chosen SEQUENCE_LENGTH")

    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    test_size = total_samples - train_size - val_size
    print(f"Samples -> train: {train_size}, val: {val_size}, test: {test_size}")

    # Prepare sequences
    X, y = prepare_sequences(scaled_consumption, SEQUENCE_LENGTH)
    X_train = X[:train_size]; y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]; y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]; y_test = y[train_size+val_size:]

    # Reshape (already (N, seq_len))
    # Build datasets/dataloaders
    train_ds = SequenceDataset(X_train.reshape(-1, SEQUENCE_LENGTH, 1), y_train)
    val_ds = SequenceDataset(X_val.reshape(-1, SEQUENCE_LENGTH, 1), y_val)
    test_ds = SequenceDataset(X_test.reshape(-1, SEQUENCE_LENGTH, 1), y_test)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False)

    print("Building model...")
    model = ComplexLSTM(SEQUENCE_LENGTH, lstm_units=CONFIG['LSTM_UNITS'], dropout_rate=CONFIG['DROPOUT_RATE'])
    print(model)

    try:
        model, history = train_model(model, train_loader, val_loader, CONFIG, device)
    except Exception as e:
        print("Training error:", e)
        print("Attempting CPU fallback model with smaller size...")
        model = ComplexLSTM(SEQUENCE_LENGTH, lstm_units=min(CONFIG['LSTM_UNITS'], 64), dropout_rate=min(CONFIG['DROPOUT_RATE'], 0.2))
        model, history = train_model(model, train_loader, val_loader, CONFIG, device)

    # Save final model
    final_model_path = os.path.join(CONFIG['MODEL_DIR'], 'electricity_complex_lstm_model_final.pt')
    torch.save({'model_state': model.state_dict(), 'scaler': scaler}, final_model_path)
    print("Final model saved to", final_model_path)

    # Evaluate on train/val/test
    def predict_dataset(model, loader, device):
        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                if USE_AMP:
                    with torch.cuda.amp.autocast():
                        p = model(xb).cpu().numpy().ravel()
                else:
                    p = model(xb).cpu().numpy().ravel()
                preds.append(p)
                trues.append(yb.cpu().numpy().ravel())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        return preds, trues

    train_pred_scaled, train_true_scaled = predict_dataset(model, train_loader, device)
    val_pred_scaled, val_true_scaled = predict_dataset(model, val_loader, device)
    test_pred_scaled, test_true_scaled = predict_dataset(model, test_loader, device)

    # inverse scale
    train_pred = scaler.inverse_transform(train_pred_scaled.reshape(-1,1)).flatten()
    val_pred = scaler.inverse_transform(val_pred_scaled.reshape(-1,1)).flatten()
    test_pred = scaler.inverse_transform(test_pred_scaled.reshape(-1,1)).flatten()

    train_true = scaler.inverse_transform(train_true_scaled.reshape(-1,1)).flatten()
    val_true = scaler.inverse_transform(val_true_scaled.reshape(-1,1)).flatten()
    test_true = scaler.inverse_transform(test_true_scaled.reshape(-1,1)).flatten()

    train_metrics = calculate_metrics(train_true, train_pred, "Training")
    val_metrics = calculate_metrics(val_true, val_pred, "Validation")
    test_metrics = calculate_metrics(test_true, test_pred, "Test")

    # Forecast future
    print(f"\nGenerating {FORECAST_HORIZON//24} day forecast ({FORECAST_HORIZON} hours)...")
    last_sequence = scaled_consumption[-SEQUENCE_LENGTH:]
    future_predictions = forecast_future(model, last_sequence, FORECAST_HORIZON, scaler, device)

    last_date = df['DateTime'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=FORECAST_HORIZON, freq='H')
    forecast_df = pd.DataFrame({'DateTime': future_dates, 'predicted_consumption': future_predictions})
    forecast_csv = f'electricity_forecast_{FORECAST_HORIZON//24}days.csv'
    forecast_df.to_csv(forecast_csv, index=False)
    print("Forecast saved to", forecast_csv)

    # Plots: training history
    plt.figure(figsize=(12,6))
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.title('Training / Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Test results visualization
    recent_range = min(500, len(test_true))
    if len(test_true) > 0:
        plt.figure(figsize=(14,5))
        plt.plot(test_true[-recent_range:], label='Actual', linewidth=1)
        plt.plot(test_pred[-recent_range:], label='Predicted', linewidth=1)
        plt.title(f"Test Data - Actual vs Predicted (Last {recent_range} hours)")
        plt.xlabel('Time Steps')
        plt.ylabel('Consumption (kWh)')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Future forecast plot
    plt.figure(figsize=(14,5))
    plt.plot(future_predictions, linewidth=1.5, alpha=0.8)
    plt.title(f'Electricity Consumption Forecast ({FORECAST_HORIZON//24:.0f} Days / {FORECAST_HORIZON} Hours)')
    plt.xlabel('Hours from Now')
    plt.ylabel('Consumption (kWh)')
    plt.grid(True)
    if FORECAST_HORIZON >= 168:
        for week in range(int(FORECAST_HORIZON/168) + 1):
            plt.axvline(x=week*168, color='red', linestyle='--', alpha=0.6, linewidth=1)
            if week < int(FORECAST_HORIZON/168):
                plt.text(week*168 + 84, plt.ylim()[1]*0.95, f'Week {week+1}', ha='center', va='top', fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Save history / metrics
    pd.DataFrame(history).to_csv('training_history.csv', index=False)
    metrics_summary = {
        'Dataset': ['Training', 'Validation', 'Test'],
        'MAE': [train_metrics['MAE'], val_metrics['MAE'], test_metrics['MAE']],
        'RMSE': [train_metrics['RMSE'], val_metrics['RMSE'], test_metrics['RMSE']],
        'MAPE': [train_metrics['MAPE'], val_metrics['MAPE'], test_metrics['MAPE']],
        'R2': [train_metrics['R2'], val_metrics['R2'], test_metrics['R2']]
    }
    pd.DataFrame(metrics_summary).to_csv('model_metrics.csv', index=False)
    print("\n=== Completed ===")
    print(f"Files Generated:\n - {final_model_path}\n - {forecast_csv}\n - training_history.csv\n - model_metrics.csv")
    print("\nForecast Sample (first 24 hours):")
    print(forecast_df.head(24).round(2).to_string(index=False))
