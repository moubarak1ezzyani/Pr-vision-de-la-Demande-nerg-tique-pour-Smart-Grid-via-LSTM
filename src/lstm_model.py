from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# --- LSTM : build architecture
def build_lstm_model(input_shape):
    model = Sequential([
        # -> Explicit input layer
        Input(shape=input_shape),   # input_shape = (timesteps, features) -> (24, 9)
        
        # -> LSTM layer
        LSTM(64, return_sequences=False),
        
        # overfitting : Regularization  
        Dropout(0.2),
        
        # Output layer 
        Dense(1)    # 1 neuron --> 1 consumption value predicted
    ])
    
    model.compile(optimizer="adam", loss="mse")
    return model