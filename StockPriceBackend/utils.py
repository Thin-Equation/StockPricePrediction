import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
from scipy.stats import linregress

def validate_csv(df):
    """Validate if the uploaded CSV has the required columns"""
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    return all(col in df.columns for col in required_columns)

def add_technical_indicators(df):
    """Add technical indicators to the DataFrame"""
    # Make a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Convert Date to datetime if it's not already
    if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date to ensure chronological order
    df = df.sort_values('Date')
    
    # Calculate True Range
    df['True Range'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    
    # Calculate 20-day Exponential Moving Average
    df['20 EMA'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # Calculate Bollinger Bands
    df['BB Mid'] = df['Close'].rolling(window=20).mean()
    df['BB Std'] = df['Close'].rolling(window=20).std()
    df['BB Upper'] = df['BB Mid'] + (df['BB Std'] * 2)
    df['BB Lower'] = df['BB Mid'] - (df['BB Std'] * 2)
    
    # Calculate MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD Histogram'] = df['MACD'] - df['MACD Signal']
    
    # Drop NA values resulting from calculations
    df = df.dropna()
    
    return df

def prepare_sequences(X, y, sequence_length=20):
    """Prepare sequences for LSTM model"""
    sequences = []
    next_values = []
    
    for i in range(len(X) - sequence_length):
        sequences.append(X[i:i+sequence_length])
        next_values.append(y[i+sequence_length])
    
    return np.array(sequences), np.array(next_values)

def prepare_data(df, sequence_length=20, with_indicators=True):
    """Prepare data for prediction"""
    if with_indicators:
        features = ['Open', 'High', 'Low', 'Volume', 'True Range', '20 EMA', 
                    'BB Upper', 'BB Mid', 'BB Lower', 'MACD', 'MACD Signal', 'MACD Histogram']
    else:
        features = ['Open', 'High', 'Low', 'Volume']
    
    X = df[features].values
    y = df['Close'].values
    
    # Scale the data
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    
    X = X_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y.reshape(-1, 1))
    
    # Create sequences
    X_seq, y_seq = prepare_sequences(X, y, sequence_length)
    
    # Save the scalers for later use
    os.makedirs('models', exist_ok=True)
    joblib.dump(X_scaler, 'models/X_scaler.pkl')
    joblib.dump(y_scaler, 'models/y_scaler.pkl')
    
    return X_seq, y_seq, X_scaler, y_scaler

def plot_predictions(actual_values, predicted_values, title="Stock Price Prediction", uncertainty=None):
    """Generate plot of actual vs predicted values with optional uncertainty bands"""
    plt.figure(figsize=(12, 6))
    
    # Plot actual values
    plt.plot(actual_values, label='Actual', linewidth=2)
    
    # Plot predicted values
    plt.plot(predicted_values, label='Predicted', linewidth=2, alpha=0.8)
    
    # Plot uncertainty bands if available
    if uncertainty is not None:
        uncertainty = uncertainty.reshape(-1)
        upper_bound = predicted_values.reshape(-1) + 1.96 * uncertainty
        lower_bound = predicted_values.reshape(-1) - 1.96 * uncertainty
        plt.fill_between(
            range(len(predicted_values)), 
            lower_bound, 
            upper_bound, 
            color='lightblue', 
            alpha=0.4, 
            label='95% Confidence Interval'
        )
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add timestamp
    plt.annotate(
        f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
        xy=(0.01, 0.01), 
        xycoords='figure fraction', 
        fontsize=8
    )
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    
    # Convert plot to base64 string
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return plot_base64

def calculate_metrics(y_true, y_pred):
    """Calculate performance metrics"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    return {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse)
    }

def extract_market_regime(df):
    """
    Detect market regime (bullish, bearish, or sideways)
    based on price trends and volatility
    """
    # Ensure DataFrame has the required columns
    if 'Close' not in df.columns:
        return "unknown"
    
    # Make sure we have enough data
    if len(df) < 30:
        return "insufficient_data"
    
    # Calculate short and long term trends
    df = df.copy()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    
    # Drop NaN values
    df = df.dropna()
    if len(df) < 20:
        return "insufficient_data"
    
    # Calculate recent volatility (standard deviation)
    recent_volatility = df['Close'].pct_change().rolling(window=20).std().iloc[-1]
    
    # Calculate trend using linear regression on recent prices
    y = df['Close'].values[-30:]
    x = np.arange(len(y))
    slope, _, r_value, _, _ = linregress(x, y)
    
    trend_strength = abs(r_value)
    
    # Determine market regime
    if slope > 0 and df['SMA20'].iloc[-1] > df['SMA50'].iloc[-1]:
        regime = "bullish"
        if recent_volatility > 0.015:  # High volatility
            regime += "_volatile"
    elif slope < 0 and df['SMA20'].iloc[-1] < df['SMA50'].iloc[-1]:
        regime = "bearish"
        if recent_volatility > 0.015:  # High volatility
            regime += "_volatile"
    else:
        regime = "sideways"
        if trend_strength < 0.3:
            regime += "_low_conviction"
        elif trend_strength > 0.7:
            regime += "_high_conviction"
    
    return regime