import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, QComboBox, 
                             QPushButton, QVBoxLayout, QGridLayout, QMessageBox)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib
import os

weights_path = "./weights"
cryptos = ["BTC", "ETH", "LTC", "XRP"]
seasons = ["Winter", "Summer", "Fall", "Spring"]

def build_dense_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def load_models(coin):
    nn_model = None
    nn_weights_path = f"{weights_path}/{coin}/dense_nn_weights_{coin}.weights.h5"
    
    scaler_X = None
    scaler_y = None
    
    if os.path.exists(nn_weights_path):
        try:
            scaler_X = joblib.load(f"{weights_path}/{coin}/scaler_X_{coin}.joblib")
            scaler_y = joblib.load(f"{weights_path}/{coin}/scaler_y_{coin}.joblib")
            
            input_features = scaler_X.n_features_in_
            nn_model = build_dense_model(input_features)
            nn_model.load_weights(nn_weights_path)
        except Exception as e:
            print(f"Warning: Could not load NN model for {coin}: {e}")
    
    lr_binary = joblib.load(f"{weights_path}/{coin}/lr_binary_model_{coin}.joblib")
    lr_model = joblib.load(f"{weights_path}/{coin}/lr_model_{coin}.joblib")
    
    return nn_model, lr_binary, lr_model, scaler_X, scaler_y

class CryptoPredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crypto Next-Day Predictor")
        self.setMinimumWidth(400)
        self.layout = QVBoxLayout()
        
        grid = QGridLayout()
        
        grid.addWidget(QLabel("Select Crypto:"), 0, 0)
        self.crypto_combo = QComboBox()
        self.crypto_combo.addItems(cryptos)
        grid.addWidget(self.crypto_combo, 0, 1)
        
        grid.addWidget(QLabel("Select Season:"), 1, 0)
        self.season_combo = QComboBox()
        self.season_combo.addItems(seasons)
        grid.addWidget(self.season_combo, 1, 1)
        
        grid.addWidget(QLabel("Open:"), 2, 0)
        self.entry_open = QLineEdit()
        self.entry_open.setPlaceholderText("e.g., 50000")
        grid.addWidget(self.entry_open, 2, 1)
        
        grid.addWidget(QLabel("High:"), 3, 0)
        self.entry_high = QLineEdit()
        self.entry_high.setPlaceholderText("e.g., 51000")
        grid.addWidget(self.entry_high, 3, 1)
        
        grid.addWidget(QLabel("Low:"), 4, 0)
        self.entry_low = QLineEdit()
        self.entry_low.setPlaceholderText("e.g., 49500")
        grid.addWidget(self.entry_low, 4, 1)
        
        grid.addWidget(QLabel("Close:"), 5, 0)
        self.entry_close = QLineEdit()
        self.entry_close.setPlaceholderText("e.g., 50500")
        grid.addWidget(self.entry_close, 5, 1)
        
        self.predict_button = QPushButton("Predict Next-Day Close")
        self.predict_button.setStyleSheet("font-weight: bold; padding: 10px;")
        self.predict_button.clicked.connect(self.predict)
        grid.addWidget(self.predict_button, 6, 0, 1, 2)
        
        self.layout.addLayout(grid)
        self.setLayout(self.layout)
    
    def predict(self):
        try:
            coin = self.crypto_combo.currentText()
            season = self.season_combo.currentText()
            
            if not all([self.entry_open.text(), self.entry_high.text(), 
                       self.entry_low.text(), self.entry_close.text()]):
                QMessageBox.warning(self, "Input Error", "Please fill in all price fields.")
                return
            
            current_prices = [
                float(self.entry_open.text()),
                float(self.entry_high.text()),
                float(self.entry_low.text()),
                float(self.entry_close.text())
            ]
            
            nn_model, lr_binary, lr_model, scaler_X, scaler_y = load_models(coin)
            
            result_text = f"=== {coin} Next-Day Predictions ===\n\n"
            
            crypto_encoding = [1 if c == coin else 0 for c in cryptos]
            season_encoding = [1 if s == season else 0 for s in seasons]
            
            lr_features = [current_prices[0], current_prices[1], current_prices[2], current_prices[3]] + crypto_encoding + season_encoding
            X_lr = np.array(lr_features).reshape(1, -1)
            
            lr_pred = lr_model.predict(X_lr)[0]
            result_text += f"Linear Regression: ${lr_pred:.2f}\n\n"
            
            binary_pred = lr_binary.predict(X_lr)[0]
            binary_label = "UP" if binary_pred == 1 else "DOWN"
            current_close = current_prices[3]
            result_text += f"Movement: {binary_label}\n"
            result_text += f"(Current close: ${current_close:.2f})\n\n"
            
            if nn_model is not None and scaler_X is not None and scaler_y is not None:
                nn_features = [current_prices[0], current_prices[1], current_prices[2], current_prices[3]] + crypto_encoding + season_encoding
                X_nn = np.array(nn_features).reshape(1, -1)
                X_scaled = scaler_X.transform(X_nn)
                nn_pred_scaled = nn_model.predict(X_scaled, verbose=0)
                nn_pred = scaler_y.inverse_transform(nn_pred_scaled.reshape(-1, 1))[0][0]
                result_text += f"Neural Network: ${nn_pred:.2f}\n"
            else:
                result_text += "Neural Network: Not available\n"
            
            QMessageBox.information(self, "Prediction Results", result_text)
            
        except ValueError as ve:
            QMessageBox.critical(self, "Input Error", f"Please enter valid numbers.\n{str(ve)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CryptoPredictor()
    window.show()
    sys.exit(app.exec())