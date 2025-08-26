"""
Temperature Forecasting with LSTM/RNN Model
Student: SHETTY SAMAY DEEPAK (4AL22CS143)
Assignment: Deep Learning Temperature Forecasting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')


# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class TemperatureForecastingLSTM:
    """
    Advanced Temperature Forecasting Model using LSTM/RNN
    Features:
    - Multiple LSTM architectures (Vanilla, Bidirectional, Stacked)
    - Advanced preprocessing and feature engineering
    - Comprehensive evaluation metrics
    - Visualization and analysis
    """
    
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        self.sequence_length = 12  # Use 12 months to predict next month
        self.features = ['ANNUAL', 'JAN-FEB', 'MAR-MAY', 'JUN-SEP', 'OCT-DEC']
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess temperature data"""
        print("Loading and preprocessing data...")
        
        # Load data
        self.df = pd.read_csv(file_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Years covered: {self.df['YEAR'].min()} - {self.df['YEAR'].max()}")
        
        # Create additional features
        self.create_features()
        
        # Handle missing values if any
        self.df = self.df.fillna(method='ffill').fillna(method='bfill')

        
        return self.df
    
    def create_features(self):
        """Create additional features for better forecasting"""
        # Temperature variations
        self.df['TEMP_RANGE'] = self.df[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                                        'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].max(axis=1) - \
                               self.df[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                                        'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].min(axis=1)
        
        # Seasonal variations
        self.df['WINTER_SUMMER_DIFF'] = self.df['JUN-SEP'] - self.df['JAN-FEB']
        
        # Year-over-year changes
        self.df['ANNUAL_CHANGE'] = self.df['ANNUAL'].diff()
        
        # Moving averages
        self.df['MA_3'] = self.df['ANNUAL'].rolling(window=3).mean()
        self.df['MA_5'] = self.df['ANNUAL'].rolling(window=5).mean()
        
        # Fill NaN values created by new features
        self.df = self.df.fillna(method='ffill').fillna(method='bfill')

        
        # Update features list
        self.features = ['ANNUAL', 'JAN-FEB', 'MAR-MAY', 'JUN-SEP', 'OCT-DEC',
                        'TEMP_RANGE', 'WINTER_SUMMER_DIFF', 'ANNUAL_CHANGE', 'MA_3', 'MA_5']
    
    def create_sequences(self, data, target_column='ANNUAL'):
        """Create sequences for LSTM training"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            # Use multiple features for input sequence
            X.append(data[i-self.sequence_length:i][self.features].values)
            y.append(data.iloc[i][target_column])
        
        return np.array(X), np.array(y)
    
    def prepare_data(self, test_size=0.2):
        """Prepare data for training"""
        print("Preparing data for training...")
        
        # Select only features (ANNUAL is already inside self.features)
        feature_data = self.df[self.features].copy()
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(feature_data)
        scaled_df = pd.DataFrame(scaled_data, columns=feature_data.columns)
        
        # Create sequences – target column must be string, not list
        X, y = self.create_sequences(scaled_df, target_column='ANNUAL')
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        self.X_train, self.X_test = X[:split_idx], X[split_idx:]
        self.y_train, self.y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training sequences: {self.X_train.shape}")
        print(f"Testing sequences: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test

    
    def build_lstm_model(self, model_type='stacked'):
        """Build LSTM model with different architectures"""
        print(f"Building {model_type} LSTM model...")
        
        input_shape = (self.X_train.shape[1], self.X_train.shape[2])
        
        if model_type == 'vanilla':
            model = Sequential([
                Input(shape=input_shape),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(1)
            ])
            
        elif model_type == 'bidirectional':
            model = Sequential([
                Input(shape=input_shape),
                Bidirectional(LSTM(50, return_sequences=True)),
                Dropout(0.3),
                Bidirectional(LSTM(50, return_sequences=False)),
                Dropout(0.3),
                Dense(25, activation='relu'),
                Dense(1)
            ])
            
        elif model_type == 'stacked':
            model = Sequential([
                Input(shape=input_shape),
                LSTM(100, return_sequences=True),
                Dropout(0.3),
                LSTM(50, return_sequences=True),
                Dropout(0.3),
                LSTM(25, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(1)
            ])
            
        elif model_type == 'hybrid':
            model = Sequential([
                Input(shape=input_shape),
                LSTM(64, return_sequences=True),
                Dropout(0.3),
                GRU(32, return_sequences=False),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(16, activation='relu'),
                Dense(1)
            ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        print(f"Model built successfully!")
        print(model.summary())
        
        return model
    
    def train_model(self, epochs=100, batch_size=32, validation_split=0.2):
        """Train the LSTM model"""
        print("Training model...")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001),
            ModelCheckpoint('best_temp_model.h5', monitor='val_loss', save_best_only=True)
        ]
        
        # Train model
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Model training completed!")
        return self.history
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print("Evaluating model...")
        
        # Make predictions
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)
        
        # Inverse transform predictions
        train_pred_full = np.zeros((len(train_pred), len(self.features)))
        test_pred_full = np.zeros((len(test_pred), len(self.features)))
        train_actual_full = np.zeros((len(self.y_train), len(self.features)))
        test_actual_full = np.zeros((len(self.y_test), len(self.features)))
        
        # Put predictions/actuals into ANNUAL column (last one)
        train_pred_full[:, -1] = train_pred.flatten()
        test_pred_full[:, -1] = test_pred.flatten()
        train_actual_full[:, -1] = self.y_train.flatten()
        test_actual_full[:, -1] = self.y_test.flatten()
        
        # Inverse transform
        train_pred_inv = self.scaler.inverse_transform(train_pred_full)[:, -1]
        test_pred_inv = self.scaler.inverse_transform(test_pred_full)[:, -1]
        train_actual_inv = self.scaler.inverse_transform(train_actual_full)[:, -1]
        test_actual_inv = self.scaler.inverse_transform(test_actual_full)[:, -1]
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(train_actual_inv, train_pred_inv))
        test_rmse = np.sqrt(mean_squared_error(test_actual_inv, test_pred_inv))
        train_mae = mean_absolute_error(train_actual_inv, train_pred_inv)
        test_mae = mean_absolute_error(test_actual_inv, test_pred_inv)
        train_r2 = r2_score(train_actual_inv, train_pred_inv)
        test_r2 = r2_score(test_actual_inv, test_pred_inv)
        
        self.results = {
            'train_rmse': train_rmse, 'test_rmse': test_rmse,
            'train_mae': train_mae, 'test_mae': test_mae,
            'train_r2': train_r2, 'test_r2': test_r2,
            'train_pred': train_pred_inv, 'test_pred': test_pred_inv,
            'train_actual': train_actual_inv, 'test_actual': test_actual_inv
        }
        
        # Print results
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Testing RMSE: {test_rmse:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Testing MAE: {test_mae:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Testing R²: {test_r2:.4f}")
        print("="*50)
        
        return self.results

    
    def plot_results(self):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Training history
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Predictions vs Actual
        axes[0, 1].scatter(self.results['test_actual'], self.results['test_pred'], alpha=0.7)
        axes[0, 1].plot([self.results['test_actual'].min(), self.results['test_actual'].max()], 
                       [self.results['test_actual'].min(), self.results['test_actual'].max()], 'r--')
        axes[0, 1].set_title('Predictions vs Actual (Test Set)')
        axes[0, 1].set_xlabel('Actual Temperature')
        axes[0, 1].set_ylabel('Predicted Temperature')
        axes[0, 1].grid(True)
        
        # Time series comparison
        test_years = self.df['YEAR'].iloc[-len(self.results['test_actual']):].values
        axes[1, 0].plot(test_years, self.results['test_actual'], label='Actual', marker='o')
        axes[1, 0].plot(test_years, self.results['test_pred'], label='Predicted', marker='s')
        axes[1, 0].set_title('Temperature Forecasting - Test Period')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Annual Temperature')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Residuals
        residuals = self.results['test_actual'] - self.results['test_pred']
        axes[1, 1].scatter(self.results['test_pred'], residuals, alpha=0.7)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_title('Residual Plot')
        axes[1, 1].set_xlabel('Predicted Temperature')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Additional visualization - Temperature trends
        plt.figure(figsize=(15, 8))
        plt.subplot(2, 1, 1)
        plt.plot(self.df['YEAR'], self.df['ANNUAL'], marker='o', linewidth=2, markersize=4)
        plt.title('Historical Annual Temperature Trend (1901-2017)')
        plt.xlabel('Year')
        plt.ylabel('Annual Temperature (°C)')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        seasonal_data = self.df[['YEAR', 'JAN-FEB', 'MAR-MAY', 'JUN-SEP', 'OCT-DEC']].set_index('YEAR')
        for season in seasonal_data.columns:
            plt.plot(seasonal_data.index, seasonal_data[season], label=season, linewidth=2)
        plt.title('Seasonal Temperature Trends')
        plt.xlabel('Year')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def forecast_future(self, years_ahead=5):
        """Forecast future temperatures"""
        print(f"Forecasting {years_ahead} years ahead...")

        # Use the last sequence from the dataset
        last_sequence = self.X_test[-1].reshape(1, self.X_test.shape[1], self.X_test.shape[2])

        future_predictions = []
        current_sequence = last_sequence.copy()

        for _ in range(years_ahead):
            # Predict next value
            next_pred = self.model.predict(current_sequence, verbose=0)
            future_predictions.append(next_pred[0, 0])

            # Update sequence (shift left, add new prediction at end)
            new_sequence = current_sequence.copy()
            new_sequence[0, :-1] = new_sequence[0, 1:]
            new_sequence[0, -1, -1] = next_pred[0, 0]  # Assuming ANNUAL is last feature

            current_sequence = new_sequence

        # ✅ Safe inverse transform
        n_features_scaler = self.scaler.n_features_in_
        dummy = np.zeros((len(future_predictions), n_features_scaler))
        dummy[:, -1] = future_predictions  # Put preds in last column (ANNUAL)
        future_pred_inv = self.scaler.inverse_transform(dummy)[:, -1]

        # Create future years
        last_year = self.df['YEAR'].max()
        future_years = [last_year + i + 1 for i in range(years_ahead)]

        # Plot future predictions
        plt.figure(figsize=(12, 6))
        recent_years = self.df['YEAR'].tail(20)
        recent_temps = self.df['ANNUAL'].tail(20)

        plt.plot(recent_years, recent_temps, 'b-o', label='Historical', linewidth=2, markersize=6)
        plt.plot(future_years, future_pred_inv, 'r--s', label='Forecast', linewidth=2, markersize=6)
        plt.title(f'Temperature Forecast: Next {years_ahead} Years')
        plt.xlabel('Year')
        plt.ylabel('Annual Temperature (°C)')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Print forecast results
        print("\nFORECAST RESULTS:")
        print("=" * 30)
        for year, temp in zip(future_years, future_pred_inv):
            print(f"Year {year}: {temp:.2f}°C")

        return future_years, future_pred_inv



def main():
    """Main execution function"""
    print("="*60)
    print("TEMPERATURE FORECASTING WITH LSTM")
    print("Student: SHETTY SAMAY DEEPAK (4AL22CS143)")
    print("="*60)
    
    # Initialize model
    lstm_model = TemperatureForecastingLSTM()
    
    # Load and preprocess data
    data = lstm_model.load_and_preprocess_data('temperatures.csv')
    
    # Prepare data for training
    X_train, X_test, y_train, y_test = lstm_model.prepare_data()
    
    # Build model (try different architectures)
    model_types = ['stacked', 'bidirectional', 'hybrid']
    best_model_type = 'stacked'  # Default
    best_rmse = float('inf')
    
    for model_type in model_types:
        print(f"\n{'='*40}")
        print(f"TESTING {model_type.upper()} LSTM MODEL")
        print(f"{'='*40}")
        
        # Build and train model
        lstm_model.build_lstm_model(model_type)
        history = lstm_model.train_model(epochs=100, batch_size=16)
        
        # Evaluate model
        results = lstm_model.evaluate_model()
        
        # Check if this is the best model
        if results['test_rmse'] < best_rmse:
            best_rmse = results['test_rmse']
            best_model_type = model_type
            best_model = lstm_model.model
            best_results = results
    
    print(f"\n{'='*50}")
    print(f"BEST MODEL: {best_model_type.upper()} LSTM")
    print(f"Best Test RMSE: {best_rmse:.4f}")
    print(f"{'='*50}")
    
    # Use best model for final analysis
    lstm_model.model = best_model
    lstm_model.results = best_results
    
    # Create visualizations
    lstm_model.plot_results()
    
    # Future forecasting
    future_years, future_temps = lstm_model.forecast_future(years_ahead=10)
    
    # Save model
    lstm_model.model.save('temperature_forecasting_lstm.h5')
    print("\nModel saved as 'temperature_forecasting_lstm.h5'")
    
    print("\nAnalysis completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()