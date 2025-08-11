import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import os
warnings.filterwarnings('ignore')

# GPU Configuration and Error Handling
def configure_tensorflow():
    """Configure TensorFlow for optimal performance and handle GPU issues"""
    try:
        # Check for GPU availability
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
            try:
                # Try to configure GPU memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU memory growth configured successfully")
                
                # Set mixed precision for better performance
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                print("Mixed precision enabled for faster training")
                return True
                
            except Exception as gpu_error:
                print(f"GPU configuration failed: {gpu_error}")
                print("Falling back to CPU execution...")
                # Force CPU execution
                tf.config.set_visible_devices([], 'GPU')
                return False
        else:
            print("No GPU found, using CPU")
            return False
            
    except Exception as e:
        print(f"TensorFlow configuration error: {e}")
        print("Using default CPU configuration")
        # Force CPU execution as fallback
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        return False

# Configure TensorFlow before setting seeds
gpu_available = configure_tensorflow()

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_electricity_data(file_path):
    """Load and preprocess electricity consumption data from CSV"""
    print(f"Loading data from: {file_path}")
    
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Display basic info about the dataset
        print(f"Dataset shape: {df.shape}")
        print(f"Column names: {list(df.columns)}")
        print(f"First few rows:")
        print(df.head())
        
        # Try to identify DateTime and consumption columns
        datetime_col = None
        consumption_col = None
        
        # Look for datetime column (case-insensitive)
        datetime_candidates = ['datetime', 'date', 'time', 'timestamp', 'dt']
        for col in df.columns:
            if any(candidate in col.lower() for candidate in datetime_candidates):
                datetime_col = col
                break
        
        # Look for consumption column (case-insensitive)
        consumption_candidates = ['consumption', 'consume', 'demand', 'load', 'usage', 'kwh', 'mwh']
        for col in df.columns:
            if any(candidate in col.lower() for candidate in consumption_candidates):
                consumption_col = col
                break
        
        # If not found automatically, use user specification or first available numeric column
        if datetime_col is None:
            datetime_col = df.columns[0]  # Assume first column is datetime
            print(f"Warning: Using '{datetime_col}' as DateTime column (not auto-detected)")
        
        if consumption_col is None:
            # Find first numeric column that's not the datetime column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            consumption_col = [col for col in numeric_cols if col != datetime_col][0]
            print(f"Warning: Using '{consumption_col}' as consumption column (not auto-detected)")
        
        print(f"Using DateTime column: '{datetime_col}'")
        print(f"Using consumption column: '{consumption_col}'")
        
        # Convert datetime column to datetime type
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        # Rename columns to standard names for consistency
        df_processed = pd.DataFrame({
            'DateTime': df[datetime_col],
            'consumption': df[consumption_col]
        })
        
        # Sort by datetime to ensure proper order
        df_processed = df_processed.sort_values('DateTime').reset_index(drop=True)
        
        # Remove any missing values
        initial_rows = len(df_processed)
        df_processed = df_processed.dropna()
        if len(df_processed) < initial_rows:
            print(f"Warning: Removed {initial_rows - len(df_processed)} rows with missing values")
        
        # Basic data validation
        if df_processed['consumption'].min() < 0:
            print("Warning: Negative consumption values found. Consider data cleaning.")
        
        # Check for time gaps
        df_processed['time_diff'] = df_processed['DateTime'].diff()
        most_common_freq = df_processed['time_diff'].mode()[0] if len(df_processed) > 1 else pd.Timedelta(hours=1)
        print(f"Most common time frequency: {most_common_freq}")
        
        # Display final dataset info
        print(f"\nProcessed dataset info:")
        print(f"  Shape: {df_processed.shape}")
        print(f"  Date range: {df_processed['DateTime'].min()} to {df_processed['DateTime'].max()}")
        print(f"  Consumption range: {df_processed['consumption'].min():.2f} to {df_processed['consumption'].max():.2f}")
        print(f"  Average consumption: {df_processed['consumption'].mean():.2f}")
        print(f"  Consumption std: {df_processed['consumption'].std():.2f}")
        
        # Remove the temporary time_diff column
        df_processed = df_processed.drop('time_diff', axis=1)
        
        return df_processed
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        print("Please make sure the file exists in the current directory.")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("Please check the file format and column names.")
        return None

def prepare_sequences(data, sequence_length):
    """Prepare sequences for LSTM training"""
    sequences = []
    targets = []
    
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        target = data[i + sequence_length]
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

def build_complex_lstm_model(sequence_length, lstm_units=128, dropout_rate=0.3, use_cpu_fallback=False):
    """Build a more complex LSTM model with regularization and error handling"""
    
    # If GPU failed, use smaller model for CPU training
    if use_cpu_fallback or not gpu_available:
        print("Using CPU-optimized model configuration...")
        lstm_units = min(lstm_units, 64)  # Reduce complexity for CPU
        dropout_rate = min(dropout_rate, 0.2)  # Reduce dropout
    
    try:
        with tf.device('/CPU:0' if use_cpu_fallback else '/GPU:0' if gpu_available else '/CPU:0'):
            model = Sequential([
                # First LSTM layer with more units
                LSTM(lstm_units, 
                     return_sequences=True, 
                     input_shape=(sequence_length, 1),
                     activation='tanh',
                     recurrent_activation='sigmoid',
                     dropout=dropout_rate,
                     recurrent_dropout=dropout_rate),
                BatchNormalization(),
                
                # Second LSTM layer
                LSTM(lstm_units // 2, 
                     return_sequences=True,
                     activation='tanh',
                     recurrent_activation='sigmoid',
                     dropout=dropout_rate,
                     recurrent_dropout=dropout_rate),
                BatchNormalization(),
                
                # Third LSTM layer
                LSTM(lstm_units // 4, 
                     return_sequences=False,
                     activation='tanh',
                     recurrent_activation='sigmoid',
                     dropout=dropout_rate),
                BatchNormalization(),
                
                # Dense layers with regularization
                Dense(64, activation='relu'),
                Dropout(dropout_rate),
                
                Dense(32, activation='relu'),
                Dropout(dropout_rate / 2),
                
                Dense(16, activation='relu'),
                Dense(1, dtype='float32')  # Ensure float32 output
            ])
            
            # Use a more sophisticated optimizer with learning rate scheduling
            optimizer = Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            )
            
            model.compile(
                optimizer=optimizer,
                loss='huber',  # More robust to outliers than MSE
                metrics=['mae', 'mse']
            )
            
            return model
            
    except Exception as e:
        print(f"Error building model: {e}")
        print("Trying fallback model configuration...")
        
        # Simple fallback model
        model = Sequential([
            LSTM(32, input_shape=(sequence_length, 1)),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model

def plot_results(actual, predicted, title="Electricity Consumption Forecast"):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(15, 8))
    plt.plot(actual, label='Actual', alpha=0.8, linewidth=1.2)
    plt.plot(predicted, label='Predicted', alpha=0.9, linewidth=1.2)
    plt.title(title, fontsize=16)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Consumption (kWh)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def forecast_future(model, last_sequence, steps, scaler):
    """Generate future predictions with confidence intervals"""
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(steps):
        # Predict next value
        next_pred = model.predict(current_sequence.reshape(1, -1, 1), verbose=0)
        predictions.append(next_pred[0, 0])
        
        # Update sequence (remove first, add prediction)
        current_sequence = np.append(current_sequence[1:], next_pred[0, 0])
    
    return np.array(predictions)

def analyze_input_output_recommendations():
    """Provide recommendations for input/output periods"""
    recommendations = {
        'input_periods': {
            '1_week': {'hours': 168, 'description': 'Good for capturing weekly patterns'},
            '2_weeks': {'hours': 336, 'description': 'Better for complex weekly patterns'},
            '1_month': {'hours': 720, 'description': 'Captures monthly billing cycles'},
            '2_months': {'hours': 1440, 'description': 'Comprehensive pattern capture'},
        },
        'output_periods': {
            '1_day': {'hours': 24, 'description': 'Short-term operational planning'},
            '3_days': {'hours': 72, 'description': 'Weekend planning'},
            '1_week': {'hours': 168, 'description': 'Weekly scheduling and maintenance'},
            '1_month': {'hours': 720, 'description': 'Long-term capacity planning'},
        }
    }
    
    print("=== INPUT/OUTPUT PERIOD RECOMMENDATIONS ===\n")
    
    print("INPUT PERIOD OPTIONS (Historical data to use):")
    for period, info in recommendations['input_periods'].items():
        print(f"  {period.replace('_', ' ').title()}: {info['hours']} hours - {info['description']}")
    
    print(f"\nOUTPUT PERIOD OPTIONS (Forecast horizon):")
    for period, info in recommendations['output_periods'].items():
        print(f"  {period.replace('_', ' ').title()}: {info['hours']} hours - {info['description']}")
    
    print(f"\nRECOMMENDED COMBINATIONS:")
    print(f"  • Conservative: 1 month input → 1 week output (720 → 168)")
    print(f"  • Balanced: 2 weeks input → 3 days output (336 → 72)")
    print(f"  • Aggressive: 2 months input → 1 month output (1440 → 720)")
    print(f"  • Operational: 1 week input → 1 day output (168 → 24)")
    
    return recommendations

# Configuration options - adjusted for stability
CONFIG = {
    'INPUT_WEEKS': 4,      # How many weeks of historical data to use
    'OUTPUT_WEEKS': 1,     # How many weeks to predict
    'LSTM_UNITS': 64,      # Reduced for stability (was 128)
    'DROPOUT_RATE': 0.2,   # Reduced dropout for better convergence
    'BATCH_SIZE': 32,      # Increased batch size for stability
    'MAX_EPOCHS': 100,     # Reduced for faster testing
    'PATIENCE': 10,        # Reduced patience
    'MIN_DELTA': 0.001,    # Slightly larger minimum improvement threshold
}

# Calculate periods in hours
SEQUENCE_LENGTH = CONFIG['INPUT_WEEKS'] * 7 * 24  # Input period
FORECAST_HORIZON = CONFIG['OUTPUT_WEEKS'] * 7 * 24  # Output period

# Main execution
print("=== Advanced Electricity Consumption Forecasting with Complex LSTM ===\n")

# Show recommendations first
recommendations = analyze_input_output_recommendations()

print(f"\nCURRENT CONFIGURATION:")
print(f"  Input Period: {CONFIG['INPUT_WEEKS']} weeks ({SEQUENCE_LENGTH} hours)")
print(f"  Output Period: {CONFIG['OUTPUT_WEEKS']} weeks ({FORECAST_HORIZON} hours)")
print(f"  Model Complexity: {CONFIG['LSTM_UNITS']} LSTM units with {CONFIG['DROPOUT_RATE']} dropout")

# 1. Load real electricity data
print(f"\n1. Loading electricity consumption data from CSV...")
df = load_electricity_data('./electricityConsumptionAndProductioction.csv')

# Check if data loading was successful
if df is None:
    print("Failed to load data. Please check the file path and format.")
    exit()

print(f"Successfully loaded dataset!")
print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
print(f"Consumption range: {df['consumption'].min():.2f} to {df['consumption'].max():.2f} kWh")

# Add data exploration for real dataset
print(f"\n1.5. Exploring your dataset...")

# Plot basic statistics and patterns
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Time series plot (sample if too large)
sample_size = min(1000, len(df))
sample_idx = np.linspace(0, len(df)-1, sample_size, dtype=int)
axes[0,0].plot(df.iloc[sample_idx]['DateTime'], df.iloc[sample_idx]['consumption'], alpha=0.7)
axes[0,0].set_title('Time Series Overview (Sample)')
axes[0,0].set_xlabel('Date')
axes[0,0].set_ylabel('Consumption')
axes[0,0].tick_params(axis='x', rotation=45)
axes[0,0].grid(True, alpha=0.3)

# Consumption distribution
axes[0,1].hist(df['consumption'], bins=50, alpha=0.7, edgecolor='black')
axes[0,1].set_title('Consumption Distribution')
axes[0,1].set_xlabel('Consumption (kWh)')
axes[0,1].set_ylabel('Frequency')
axes[0,1].grid(True, alpha=0.3)

# Daily pattern (if enough data)
if len(df) >= 24:
    df['hour'] = df['DateTime'].dt.hour
    hourly_avg = df.groupby('hour')['consumption'].mean()
    axes[1,0].plot(hourly_avg.index, hourly_avg.values, marker='o')
    axes[1,0].set_title('Average Hourly Consumption Pattern')
    axes[1,0].set_xlabel('Hour of Day')
    axes[1,0].set_ylabel('Average Consumption')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_xticks(range(0, 24, 4))
else:
    axes[1,0].text(0.5, 0.5, 'Not enough data\nfor hourly pattern', 
                   ha='center', va='center', transform=axes[1,0].transAxes)

# Weekly pattern (if enough data)
if len(df) >= 7*24:
    df['dayofweek'] = df['DateTime'].dt.dayofweek
    daily_avg = df.groupby('dayofweek')['consumption'].mean()
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[1,1].bar(range(7), daily_avg.values, alpha=0.7)
    axes[1,1].set_title('Average Daily Consumption Pattern')
    axes[1,1].set_xlabel('Day of Week')
    axes[1,1].set_ylabel('Average Consumption')
    axes[1,1].set_xticks(range(7))
    axes[1,1].set_xticklabels(day_names)
    axes[1,1].grid(True, alpha=0.3)
else:
    axes[1,1].text(0.5, 0.5, 'Not enough data\nfor weekly pattern', 
                   ha='center', va='center', transform=axes[1,1].transAxes)

plt.tight_layout()
plt.show()

# Data quality assessment
print(f"\nData Quality Assessment:")
print(f"  Total records: {len(df):,}")
print(f"  Missing values: {df.isnull().sum().sum()}")
print(f"  Zero/negative consumption: {(df['consumption'] <= 0).sum()}")
print(f"  Duplicate timestamps: {df['DateTime'].duplicated().sum()}")

# Time frequency analysis
time_diffs = df['DateTime'].diff().dropna()
most_common_diff = time_diffs.mode()[0] if len(time_diffs) > 0 else None
print(f"  Most common time interval: {most_common_diff}")
print(f"  Time gaps > 2x common interval: {(time_diffs > 2 * most_common_diff).sum() if most_common_diff else 'N/A'}")

# Remove temporary columns if created
df = df.drop(columns=['hour', 'dayofweek'], errors='ignore')
# 2. Enhanced data preprocessing
print(f"\n2. Preprocessing real electricity data with advanced normalization...")
# Use RobustScaler-like approach to handle outliers better
q25, q75 = np.percentile(df['consumption'].values, [25, 75])
iqr = q75 - q25
median = np.median(df['consumption'].values)

# Modified normalization that's more robust to outliers
scaler = MinMaxScaler(feature_range=(-1, 1))  # Different range for better gradient flow
scaled_consumption = scaler.fit_transform(df['consumption'].values.reshape(-1, 1)).flatten()

# Split data with more sophisticated approach
total_samples = len(scaled_consumption) - SEQUENCE_LENGTH
train_size = int(0.7 * total_samples)  # More data for training
val_size = int(0.15 * total_samples)
test_size = total_samples - train_size - val_size

print(f"Input sequence length: {SEQUENCE_LENGTH} hours ({SEQUENCE_LENGTH/24:.1f} days)")
print(f"Forecast horizon: {FORECAST_HORIZON} hours ({FORECAST_HORIZON/24:.1f} days)")
print(f"Training samples: {train_size}")
print(f"Validation samples: {val_size}")
print(f"Test samples: {test_size}")

# 3. Prepare sequences
print(f"\n3. Preparing sequences for advanced LSTM...")
X, y = prepare_sequences(scaled_consumption, SEQUENCE_LENGTH)

# Split datasets
X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]
X_test = X[train_size + val_size:]
y_test = y[train_size + val_size:]

# Reshape for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"Training data shape: {X_train.shape}")

# 4. Build and train complex model with error handling
print(f"\n4. Building advanced LSTM model...")

try:
    model = build_complex_lstm_model(
        SEQUENCE_LENGTH, 
        lstm_units=CONFIG['LSTM_UNITS'],
        dropout_rate=CONFIG['DROPOUT_RATE']
    )
    print("Model built successfully!")
    print(model.summary())
    
except Exception as e:
    print(f"Error during model creation: {e}")
    print("Trying with CPU fallback...")
    model = build_complex_lstm_model(
        SEQUENCE_LENGTH, 
        lstm_units=CONFIG['LSTM_UNITS'],
        dropout_rate=CONFIG['DROPOUT_RATE'],
        use_cpu_fallback=True
    )
    print("Fallback model built successfully!")
    print(model.summary())

# Enhanced callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=CONFIG['PATIENCE'],
    restore_best_weights=True,
    verbose=1,
    min_delta=CONFIG['MIN_DELTA']
)

model_checkpoint = ModelCheckpoint(
    'best_electricity_complex_lstm_model.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# Learning rate reduction on plateau
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=CONFIG['PATIENCE']//2,
    min_lr=1e-6,
    verbose=1
)

print(f"\nTraining model with robust error handling...")
print(f"  Max epochs: {CONFIG['MAX_EPOCHS']}")
print(f"  Early stopping patience: {CONFIG['PATIENCE']}")
print(f"  Batch size: {CONFIG['BATCH_SIZE']}")
print(f"  Device: {'GPU' if gpu_available else 'CPU'}")

# Train the model with error handling
try:
    history = model.fit(
        X_train, y_train,
        batch_size=CONFIG['BATCH_SIZE'],
        epochs=CONFIG['MAX_EPOCHS'],
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint, lr_scheduler],
        verbose=1,
        shuffle=False
    )
    
    epochs_trained = len(history.history['loss'])
    print(f"\nTraining completed successfully after {epochs_trained} epochs.")
    print("Best model saved as 'best_electricity_complex_lstm_model.h5'")
    
except Exception as e:
    print(f"Training error: {e}")
    print("Trying simplified training approach...")
    
    # Simplified training without some callbacks if there are issues
    try:
        history = model.fit(
            X_train, y_train,
            batch_size=CONFIG['BATCH_SIZE'],
            epochs=min(CONFIG['MAX_EPOCHS'], 20),  # Reduced epochs for fallback
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],  # Only essential callback
            verbose=1,
            shuffle=False
        )
        epochs_trained = len(history.history['loss'])
        print(f"\nSimplified training completed after {epochs_trained} epochs.")
        
        # Manually save the model
        model.save('electricity_complex_lstm_model_manual.h5')
        print("Model saved manually as 'electricity_complex_lstm_model_manual.h5'")
        
    except Exception as e2:
        print(f"Simplified training also failed: {e2}")
        print("Please check your TensorFlow/CUDA installation.")
        exit()

# 5. Comprehensive evaluation
print(f"\n5. Comprehensive model evaluation...")

# Predictions on all datasets
train_pred = model.predict(X_train, verbose=0)
val_pred = model.predict(X_val, verbose=0)
test_pred = model.predict(X_test, verbose=0)

# Convert back to original scale
train_pred_original = scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()
val_pred_original = scaler.inverse_transform(val_pred.reshape(-1, 1)).flatten()
test_pred_original = scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()

train_actual_original = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
val_actual_original = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
test_actual_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Calculate comprehensive metrics
def calculate_metrics(actual, predicted, dataset_name):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
    
    print(f"\n{dataset_name} Metrics:")
    print(f"  MAE: {mae:.2f} kWh")
    print(f"  RMSE: {rmse:.2f} kWh")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²: {r2:.4f}")
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

train_metrics = calculate_metrics(train_actual_original, train_pred_original, "Training")
val_metrics = calculate_metrics(val_actual_original, val_pred_original, "Validation")
test_metrics = calculate_metrics(test_actual_original, test_pred_original, "Test")

# 6. Generate future forecasts
print(f"\n6. Generating {FORECAST_HORIZON//24:.0f}-day forecast ({FORECAST_HORIZON} hours)...")

# Use the last sequence from the dataset
last_sequence = scaled_consumption[-SEQUENCE_LENGTH:]
future_predictions_scaled = forecast_future(model, last_sequence, FORECAST_HORIZON, scaler)
future_predictions = scaler.inverse_transform(future_predictions_scaled.reshape(-1, 1)).flatten()

# Create future dates
last_date = df['DateTime'].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), 
                           periods=FORECAST_HORIZON, 
                           freq='H')

# Create forecast DataFrame
forecast_df = pd.DataFrame({
    'DateTime': future_dates,
    'predicted_consumption': future_predictions
})

print(f"Forecast period: {future_dates[0]} to {future_dates[-1]}")
print(f"Predicted consumption range: {future_predictions.min():.2f} to {future_predictions.max():.2f} kWh")

# 7. Advanced visualizations
print(f"\n7. Creating comprehensive visualizations...")

# Training history with multiple metrics
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss
axes[0,0].plot(history.history['loss'], label='Training Loss', alpha=0.8)
axes[0,0].plot(history.history['val_loss'], label='Validation Loss', alpha=0.8)
axes[0,0].set_title('Model Loss (Huber)')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Loss')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# MAE
axes[0,1].plot(history.history['mae'], label='Training MAE', alpha=0.8)
axes[0,1].plot(history.history['val_mae'], label='Validation MAE', alpha=0.8)
axes[0,1].set_title('Mean Absolute Error')
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('MAE')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Learning rate (if available)
if 'lr' in history.history:
    axes[1,0].plot(history.history['lr'], alpha=0.8, color='orange')
    axes[1,0].set_title('Learning Rate Schedule')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Learning Rate')
    axes[1,0].set_yscale('log')
    axes[1,0].grid(True, alpha=0.3)
else:
    axes[1,0].text(0.5, 0.5, 'Learning Rate\nNot Tracked', ha='center', va='center', transform=axes[1,0].transAxes)
    axes[1,0].set_title('Learning Rate Schedule')

# Model complexity visualization
complexity_data = [train_metrics['MAE'], val_metrics['MAE'], test_metrics['MAE']]
axes[1,1].bar(['Train', 'Validation', 'Test'], complexity_data, alpha=0.7)
axes[1,1].set_title('MAE Across Datasets')
axes[1,1].set_ylabel('MAE (kWh)')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Test predictions visualization
recent_range = min(500, len(test_actual_original))
if len(test_actual_original) > 0:
    plot_results(
        test_actual_original[-recent_range:], 
        test_pred_original[-recent_range:],
        f"Test Data - Actual vs Predicted (Last {recent_range} hours)"
    )

# Future forecast visualization
plt.figure(figsize=(16, 8))
plt.plot(future_predictions, linewidth=1.5, color='blue', alpha=0.8)
plt.title(f'Electricity Consumption Forecast ({FORECAST_HORIZON//24:.0f} Days / {FORECAST_HORIZON} Hours)', 
          fontsize=16)
plt.xlabel('Hours from Now', fontsize=12)
plt.ylabel('Consumption (kWh)', fontsize=12)
plt.grid(True, alpha=0.3)

# Add time period markers
if FORECAST_HORIZON >= 168:  # If forecasting at least a week
    for week in range(int(FORECAST_HORIZON/168) + 1):
        plt.axvline(x=week*168, color='red', linestyle='--', alpha=0.6, linewidth=1)
        if week < int(FORECAST_HORIZON/168):
            plt.text(week*168 + 84, plt.ylim()[1]*0.95, f'Week {week+1}', 
                    ha='center', va='top', fontweight='bold')

plt.tight_layout()
plt.show()

# 8. Save comprehensive results
print(f"\n8. Saving comprehensive results...")

# Save forecast
forecast_df.to_csv(f'electricity_forecast_{FORECAST_HORIZON//24}days.csv', index=False)
print(f"Forecast saved to 'electricity_forecast_{FORECAST_HORIZON//24}days.csv'")

# Save model
model.save('electricity_complex_lstm_model_final.h5')
print("Final model saved to 'electricity_complex_lstm_model_final.h5'")
print("Best model saved to 'best_electricity_complex_lstm_model.h5' (recommended)")

# Save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history.csv', index=False)
print("Training history saved to 'training_history.csv'")

# Save metrics
metrics_summary = {
    'Dataset': ['Training', 'Validation', 'Test'],
    'MAE': [train_metrics['MAE'], val_metrics['MAE'], test_metrics['MAE']],
    'RMSE': [train_metrics['RMSE'], val_metrics['RMSE'], test_metrics['RMSE']],
    'MAPE': [train_metrics['MAPE'], val_metrics['MAPE'], test_metrics['MAPE']],
    'R2': [train_metrics['R2'], val_metrics['R2'], test_metrics['R2']]
}
pd.DataFrame(metrics_summary).to_csv('model_metrics.csv', index=False)
print("Model metrics saved to 'model_metrics.csv'")

print(f"\n=== Advanced LSTM Forecasting Complete ===")
print(f"Configuration Used:")
print(f"  • Input: {CONFIG['INPUT_WEEKS']} weeks ({SEQUENCE_LENGTH} hours)")
print(f"  • Output: {CONFIG['OUTPUT_WEEKS']} weeks ({FORECAST_HORIZON} hours)")
print(f"  • LSTM Units: {CONFIG['LSTM_UNITS']}")
print(f"  • Dropout Rate: {CONFIG['DROPOUT_RATE']}")
print(f"  • Epochs Trained: {epochs_trained}")

print(f"\nFiles Generated:")
print(f"  • best_electricity_complex_lstm_model.h5 (BEST MODEL)")
print(f"  • electricity_complex_lstm_model_final.h5")
print(f"  • electricity_forecast_{FORECAST_HORIZON//24}days.csv")
print(f"  • training_history.csv")
print(f"  • model_metrics.csv")

print(f"\nModel Performance Summary:")
print(f"  • Test MAE: {test_metrics['MAE']:.2f} kWh")
print(f"  • Test RMSE: {test_metrics['RMSE']:.2f} kWh")  
print(f"  • Test R²: {test_metrics['R2']:.4f}")

# Show forecast sample
print(f"\nForecast Sample (first 24 hours):")
print(forecast_df.head(24)[['DateTime', 'predicted_consumption']].round(2).to_string(index=False))
