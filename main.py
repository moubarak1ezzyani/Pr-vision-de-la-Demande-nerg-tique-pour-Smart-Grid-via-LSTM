import numpy as np
import matplotlib.pyplot as plt
import os
from src.data_loader import load_and_preprocess_data, create_sequences
from src.lstm_model import build_lstm_model

# --- parameters ---
data_file = 'df_LSTM.csv' 
df_path = os.path.join('data', data_file)
window = 24

def main():
    print(f">>> Launching the Deep Learning Pipeline on {data_file}")
    
    # -> loading & Scaling (-1, 1)
    try:
        df_scaled, scaler = load_and_preprocess_data(df_path)
    except Exception as e:
        print(f"Critical error : {e}")
        return

    # -> sequence preparation
    target_idx = df_scaled.columns.get_loc("Consumption")
    X, y = create_sequences(df_scaled, target_col_index=target_idx, window_size=window)
    
    # -> Split (80/20) without shuffling
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # -> Model & Training
    model = build_lstm_model(input_shape=(window, X.shape[2]))
    
    print(">>> Starting LSTM training...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=15,              # convergence : 15 epochs are enough 
        batch_size=32,
        verbose=1
    )

    # -> Final Visualization 
    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(14, 6))
    plt.plot(y_test[:150], label="Actual (Normalized)", color='#1f77b4')
    plt.plot(y_pred[:150], label="Predicted (LSTM)", color='#ff7f0e', linestyle='--')
    plt.title(f"LSTM Results (Scaling -1 to 1) - Test RMSE: {history.history['val_loss'][-1]:.5f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show(block=True)

if __name__ == "__main__":
    main()