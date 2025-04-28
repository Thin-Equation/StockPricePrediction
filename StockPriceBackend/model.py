import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import joblib
import os
import pickle
import datetime
import copy
from utils import prepare_data, calculate_metrics, plot_predictions

# Try to import the extract_market_regime function, but make it optional
try:
    from utils import extract_market_regime
    has_market_regime = True
except ImportError:
    has_market_regime = False

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights

class DynamicLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=3, output_size=1, dropout=0.3, bidirectional=True):
        """Advanced PyTorch LSTM model with attention mechanism"""
        super(DynamicLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        
        # Main LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_size * self.directions)
        
        # Fully connected layers with residual connections
        fc_input_size = hidden_size * self.directions
        self.fc1 = nn.Linear(fc_input_size, fc_input_size // 2)
        self.fc2 = nn.Linear(fc_input_size // 2, fc_input_size // 4)
        self.fc3 = nn.Linear(fc_input_size // 4, output_size)
        
        # Additional layers
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(fc_input_size // 2)
        self.batch_norm2 = nn.BatchNorm1d(fc_input_size // 4)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """Forward pass with attention mechanism"""
        # LSTM layers
        lstm_output, _ = self.lstm(x)
        
        # Apply attention
        context_vector, attention_weights = self.attention(lstm_output)
        
        # First fully connected layer with batch normalization
        out = self.fc1(context_vector)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second fully connected layer with batch normalization
        out = self.fc2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Output layer
        out = self.fc3(out)
        
        return out, attention_weights

class StockPredictionModel:
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.model = None
        self.X_scaler = None
        self.y_scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_model = None
        self.best_loss = float('inf')
        self.training_history = []
        self.model_updates = 0
        
        # Model versioning
        self.model_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.model_path = f'models/stock_prediction_model_{self.model_timestamp}.pt'
        
        print(f"Using device: {self.device}")
    
    def build_model(self, input_size):
        """Build advanced LSTM model architecture"""
        model = DynamicLSTMModel(
            input_size=input_size, 
            hidden_size=64, 
            num_layers=3, 
            output_size=1, 
            dropout=0.3, 
            bidirectional=True
        )
            
        model.to(self.device)
        return model
    
    def train(self, df, with_indicators=True, epochs=20, batch_size=128, patience=5, validation_split=0.2):
        """Train the model with early stopping and adaptive learning rate"""
        try:
            print("[MODEL] Starting data preparation")
            # Prepare data with market regime detection
            X_seq, y_seq, X_scaler, y_scaler = prepare_data(
                df, 
                sequence_length=self.sequence_length,
                with_indicators=with_indicators
            )
            print(f"[MODEL] Data preparation complete: X shape: {X_seq.shape}, y shape: {y_seq.shape}")
            
            # Extract market regime features if available
            market_regime = "unknown"
            if with_indicators and has_market_regime:
                try:
                    market_regime = extract_market_regime(df)
                    print(f"[MODEL] Detected market regime: {market_regime}")
                except Exception as e:
                    print(f"[MODEL] Warning: Could not detect market regime: {str(e)}")
            
            # Store the scalers
            self.X_scaler = X_scaler
            self.y_scaler = y_scaler
            
            # Split data into train, validation, and test sets
            print("[MODEL] Splitting data into train, validation, and test sets")
            total_samples = len(X_seq)
            train_size = int(total_samples * 0.7)
            val_size = int(total_samples * 0.15)
            
            if total_samples < 60:  # Minimum data check
                print(f"[MODEL] Warning: Dataset is very small: {total_samples} samples")
                
            X_train = X_seq[:train_size]
            y_train = y_seq[:train_size]
            X_val = X_seq[train_size:train_size+val_size]
            y_val = y_seq[train_size:train_size+val_size]
            X_test = X_seq[train_size+val_size:]
            y_test = y_seq[train_size+val_size:]
            
            print(f"[MODEL] Train set: {X_train.shape[0]} samples")
            print(f"[MODEL] Validation set: {X_val.shape[0]} samples")
            print(f"[MODEL] Test set: {X_test.shape[0]} samples")
            
            # Convert to PyTorch tensors
            print("[MODEL] Converting to PyTorch tensors")
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            y_test_tensor = torch.FloatTensor(y_test).to(self.device)
            
            # Create data loaders
            print("[MODEL] Creating data loaders")
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Build model
            input_size = X_train.shape[2]
            print(f"[MODEL] Building model with input_size={input_size}")
            self.model = self.build_model(input_size)
            
            # Define optimizer and loss function
            optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
            
            # Learning rate scheduler
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
            
            # Loss function - Huber loss is more robust to outliers than MSE
            criterion = nn.SmoothL1Loss()
            
            # Training history
            history = {'train_loss': [], 'val_loss': []}
            best_val_loss = float('inf')
            epochs_no_improve = 0
            
            # Training loop with early stopping
            print(f"[MODEL] Starting training loop with {epochs} epochs")
            for epoch in range(epochs):
                try:
                    # Training phase
                    self.model.train()
                    train_loss = 0
                    
                    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                        try:
                            optimizer.zero_grad()
                            outputs, _ = self.model(X_batch)
                            loss = criterion(outputs, y_batch)
                            loss.backward()
                            
                            # Gradient clipping to prevent exploding gradients
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            
                            optimizer.step()
                            train_loss += loss.item()
                            
                            if batch_idx % 10 == 0:
                                print(f"[MODEL] Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
                        except Exception as batch_error:
                            print(f"[MODEL] Error in batch {batch_idx}: {str(batch_error)}")
                            import traceback
                            print(traceback.format_exc())
                            continue  # Skip this batch and continue with next one
                    
                    avg_train_loss = train_loss / len(train_loader)
                    
                    # Validation phase
                    self.model.eval()
                    val_loss = 0
                    
                    with torch.no_grad():
                        for X_batch, y_batch in val_loader:
                            outputs, _ = self.model(X_batch)
                            batch_loss = criterion(outputs, y_batch)
                            val_loss += batch_loss.item()
                    
                    avg_val_loss = val_loss / len(val_loader)
                    
                    # Update learning rate based on validation loss
                    scheduler.step(avg_val_loss)
                    
                    # Save history
                    history['train_loss'].append(avg_train_loss)
                    history['val_loss'].append(avg_val_loss)
                    
                    # Print progress
                    print(f"[MODEL] Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
                    
                    # Check for improvement
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        epochs_no_improve = 0
                        
                        # Save the best model
                        self.best_model = copy.deepcopy(self.model.state_dict())
                        print(f"[MODEL] New best model saved with validation loss: {best_val_loss:.6f}")
                    else:
                        epochs_no_improve += 1
                        print(f"[MODEL] No improvement for {epochs_no_improve} epochs")
                        
                    # Early stopping
                    if epochs_no_improve >= patience:
                        print(f"[MODEL] Early stopping triggered after {epoch+1} epochs")
                        break
                except Exception as epoch_error:
                    print(f"[MODEL] Error during epoch {epoch+1}: {str(epoch_error)}")
                    import traceback
                    print(traceback.format_exc())
                    continue  # Skip this epoch and continue with next one
            
            print("[MODEL] Training complete, evaluating model")
            
            # Load the best model for evaluation
            if self.best_model is not None:
                self.model.load_state_dict(self.best_model)
            
            # Evaluate on test set
            self.model.eval()
            try:
                with torch.no_grad():
                    print("[MODEL] Running test set evaluation")
                    y_pred, attention_weights = self.model(X_test_tensor)
                    y_pred = y_pred.cpu().numpy()
                    test_loss = criterion(self.model(X_test_tensor)[0], y_test_tensor).item()
                    print(f"[MODEL] Test loss: {test_loss:.6f}")
            except Exception as eval_error:
                print(f"[MODEL] Error during evaluation: {str(eval_error)}")
                import traceback
                print(traceback.format_exc())
                raise
            
            # Inverse transform the predictions and actual values
            print("[MODEL] Inverse transforming predictions")
            y_pred_inv = self.y_scaler.inverse_transform(y_pred)
            y_test_inv = self.y_scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # Calculate metrics
            print("[MODEL] Calculating metrics")
            metrics = calculate_metrics(y_test_inv, y_pred_inv)
            print(f"[MODEL] Metrics: {metrics}")
            
            # Generate prediction plot
            print("[MODEL] Generating prediction plot")
            plot = plot_predictions(y_test_inv, y_pred_inv, "Stock Price Prediction")
            
            # Save model for future use
            print("[MODEL] Saving model")
            os.makedirs('models', exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'input_size': input_size,
                'timestamp': self.model_timestamp,
                'metrics': metrics,
                'with_indicators': with_indicators
            }, self.model_path)
            
            # Also save as the default model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'input_size': input_size,
                'timestamp': self.model_timestamp,
                'metrics': metrics,
                'with_indicators': with_indicators
            }, 'models/stock_prediction_model.pt')
            
            # Save scalers
            joblib.dump(self.X_scaler, 'models/X_scaler.pkl')
            joblib.dump(self.y_scaler, 'models/y_scaler.pkl')
            
            # Save training history
            self.training_history.append({
                'timestamp': self.model_timestamp,
                'metrics': metrics,
                'history': history
            })
            
            # Save training history
            with open('models/training_history.pkl', 'wb') as f:
                pickle.dump(self.training_history, f)
            
            print("[MODEL] Model saving complete")
            
            return {
                'loss': float(test_loss),
                'metrics': metrics,
                'plot': plot,
                'history': history,
                'attention_weights': attention_weights.cpu().numpy().mean(axis=0).tolist()
            }
        except Exception as e:
            print(f"[MODEL] Critical error in train method: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise
    
    def update_model(self, new_data, learning_rate=0.0005, epochs=5):
        """Continuously update the model with new data"""
        if self.model is None:
            raise Exception("No model found. Please train a model first.")
            
        # Prepare the new data
        X_seq, y_seq, _, _ = prepare_data(
            new_data,
            sequence_length=self.sequence_length,
            with_indicators=True
        )
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)
        
        # Create dataset and loader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Set model to training mode
        self.model.train()
        
        # Use a smaller learning rate for fine-tuning
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.SmoothL1Loss()
        
        # Fine-tune the model
        update_history = {'loss': []}
        
        for epoch in range(epochs):
            epoch_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs, _ = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(loader)
            update_history['loss'].append(avg_loss)
            print(f"Update Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            
        # Save the updated model
        self.model_updates += 1
        update_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        update_model_path = f'models/stock_prediction_model_update_{update_timestamp}.pt'
        
        input_size = X_seq.shape[2]
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': input_size,
            'timestamp': update_timestamp,
            'update_number': self.model_updates,
            'base_model': self.model_timestamp
        }, update_model_path)
        
        # Also update the default model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': input_size,
            'timestamp': update_timestamp
        }, 'models/stock_prediction_model.pt')
        
        return {
            'message': 'Model successfully updated',
            'update_history': update_history,
            'update_number': self.model_updates
        }
    
    def predict(self, df, with_indicators=True, ensemble=False, num_samples=5):
        """Make predictions with the trained model with optional ensemble prediction"""
        if self.model is None:
            try:
                # Load model
                checkpoint = torch.load('models/stock_prediction_model.pt', map_location=self.device)
                input_size = checkpoint['input_size']
                
                # Initialize the model architecture
                self.model = self.build_model(input_size)
                
                # Load model weights
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                # Load scalers
                self.X_scaler = joblib.load('models/X_scaler.pkl')
                self.y_scaler = joblib.load('models/y_scaler.pkl')
            except Exception as e:
                raise Exception(f"Error loading model: {str(e)}")
        
        # Prepare data
        X_seq, y_seq, _, _ = prepare_data(
            df, 
            sequence_length=self.sequence_length,
            with_indicators=with_indicators
        )
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        # Make predictions
        self.model.eval()
        
        if (ensemble):
            # Ensemble prediction with Monte Carlo Dropout
            predictions = []
            attention_data = []
            
            # Enable dropout during inference for Monte Carlo Dropout
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    module.train()
                    
            # Generate multiple predictions
            for _ in range(num_samples):
                with torch.no_grad():
                    pred, attn = self.model(X_tensor)
                    predictions.append(pred.cpu().numpy())
                    attention_data.append(attn.cpu().numpy().mean(axis=0))
            
            # Average the predictions
            y_pred = np.mean(predictions, axis=0)
            
            # Calculate uncertainty
            y_std = np.std(predictions, axis=0)
            
            attention_weights = np.mean(attention_data, axis=0)
        else:
            # Single prediction
            with torch.no_grad():
                y_pred, attention_weights = self.model(X_tensor)
                y_pred = y_pred.cpu().numpy()
                attention_weights = attention_weights.cpu().numpy().mean(axis=0)
            y_std = None
        
        # Inverse transform the predictions and actual values
        y_pred_inv = self.y_scaler.inverse_transform(y_pred)
        y_test_inv = self.y_scaler.inverse_transform(y_seq.reshape(-1, 1))
        
        if y_std is not None:
            y_std_inv = y_std * (self.y_scaler.data_max_ - self.y_scaler.data_min_)
        else:
            y_std_inv = None
        
        # Calculate metrics
        metrics = calculate_metrics(y_test_inv, y_pred_inv)
        
        # Generate prediction plot
        plot = plot_predictions(y_test_inv, y_pred_inv, "Stock Price Prediction", uncertainty=y_std_inv)
        
        # Get the last sequence for future prediction
        future_price = None
        future_std = None
        
        if len(X_seq) > 0:
            last_sequence = torch.FloatTensor(X_seq[-1:]).to(self.device)
            
            if ensemble:
                future_preds = []
                for _ in range(num_samples):
                    with torch.no_grad():
                        pred, _ = self.model(last_sequence)
                        future_preds.append(pred.cpu().numpy()[0][0])
                
                future_pred_mean = np.mean(future_preds)
                future_pred_std = np.std(future_preds)
                
                future_price = self.y_scaler.inverse_transform([[future_pred_mean]])[0][0]
                future_std = future_pred_std * (self.y_scaler.data_max_ - self.y_scaler.data_min_)[0]
            else:
                with torch.no_grad():
                    future_pred, _ = self.model(last_sequence)
                    future_pred = future_pred.cpu().numpy()
                
                future_price = self.y_scaler.inverse_transform(future_pred)[0][0]
        
        result = {
            'metrics': metrics,
            'plot': plot,
            'predictions': y_pred_inv.flatten().tolist(),
            'actual': y_test_inv.flatten().tolist(),
            'next_day_prediction': float(future_price) if future_price is not None else None,
        }
        
        # Add uncertainty information if ensemble method was used
        if ensemble:
            result['uncertainty'] = {
                'prediction_std': y_std_inv.flatten().tolist() if y_std_inv is not None else None,
                'next_day_std': float(future_std) if future_std is not None else None,
                'confidence_interval_95': [
                    float(future_price - 1.96 * future_std) if future_price is not None else None,
                    float(future_price + 1.96 * future_std) if future_price is not None else None
                ]
            }
            
        # Add attention weights if available
        if attention_weights is not None:
            result['attention_weights'] = attention_weights.flatten().tolist()
        
        return result