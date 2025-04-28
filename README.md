# Stock Price Prediction

An advanced stock price prediction system with a user-friendly interface that allows users to upload their own stock data and analyze predictions.

## Features

- **CSV Upload**: Upload your own stock CSV data for analysis
- **Technical Indicators**: Automatically calculates technical indicators for better predictions
- **Advanced LSTM Model**: Uses deep learning LSTM networks with attention mechanism for accurate price predictions
- **Interactive UI**: Modern web interface built with Next.js and Tailwind CSS
- **Custom Training**: Adjust epochs and batch size to fine-tune your model
- **Visualization**: See prediction results with interactive charts
- **Ensemble Prediction**: Option for Monte Carlo dropout ensemble predictions with uncertainty estimates

## Project Structure

- **StockPriceBackend**: Python-based FastAPI server for data processing and model training
- **StockPriceFrontend**: Next.js web application with TypeScript for user interface
- **Models**: Pre-trained and user-trained models are saved for future use

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

### Installation

#### Backend Setup

1. Navigate to the backend directory:
   ```
   cd StockPriceBackend
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the FastAPI server:
   ```
   python main.py
   ```
   The backend server will start at http://localhost:8000

#### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd StockPriceFrontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Run the development server:
   ```
   npm run dev
   ```
   The frontend will start at http://localhost:3000

## Using the Application

1. Upload a CSV file with stock price data
   - Required columns: Date, Open, High, Low, Close, Volume

2. Configure model parameters:
   - Choose whether to use technical indicators
   - Adjust the number of epochs for training
   - Set batch size

3. Train your model before forecasting
   - The forecast button will be disabled until a model has been trained

4. View prediction results including:
   - Model performance metrics (MAE, MSE, RMSE)
   - Visual comparison of actual vs. predicted prices
   - Next-day price prediction

## CSV Format

Your CSV file should have the following columns:
- Date: The date of the stock price (YYYY-MM-DD format)
- Open: Opening price
- High: Highest price during the time period
- Low: Lowest price during the time period
- Close: Closing price
- Volume: Trading volume

## Model Architecture

The prediction model uses a sophisticated architecture:
- Bidirectional LSTM layers for capturing temporal patterns
- Attention mechanism to focus on the most relevant time steps
- Batch normalization and dropout for regularization
- Adaptive learning rate with early stopping

## Sample Data

The repository includes sample CSV files for testing:
- `AAPL Daily.csv`: Apple stock daily prices
- `AAPL Daily with Technical Indicators.csv`: Daily prices with pre-calculated indicators
- `AAPL Hourly.csv`: Apple stock hourly prices
- `AAPL Hourly with Technical Indicators.csv`: Hourly prices with pre-calculated indicators