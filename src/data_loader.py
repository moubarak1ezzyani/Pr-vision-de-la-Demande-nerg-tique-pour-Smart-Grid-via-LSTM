import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# --- loading data & normalization
def load_and_preprocess_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File Not Found : {filepath}")

    # -> loading data
    df = pd.read_csv(filepath)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.sort_values("DateTime").set_index("DateTime")

    # -> Normalization (-1, 1) : based on 'ml_lab'
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    # -> DataFrame : keep the column names
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    
    return df_scaled, scaler

# -> LSTM : data =>> 3D seq 
def create_sequences(data, target_col_index=0, window_size=24):
    data_values = data.values
    X, y = [], []
    
    for i in range(window_size, len(data_values)):
        X.append(data_values[i-window_size:i])          # Input: (Samples, Features)
        y.append(data_values[i, target_col_index])
        
    return np.array(X), np.array(y)             # Output X: (Samples, Window_Size, Features) 
                                                # Output y: (Samples,)