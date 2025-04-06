import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GRU, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
from sklearn.model_selection import train_test_split
import os, datetime

class DeepHedgerTF:
    def __init__(self, model, S0=100, K_strike=100, M=10**5, N=100, T=30, r=0, CALL=1):
        # Set parameters
        self.model = model
        self.S0 = S0
        self.K_strike = K_strike
        self.M = M
        self.N = N       # Number of discrete time steps
        self.T = T   # Total time horizon (e.g., 30 days)
        self.dt = T / N
        self.r = r
        self.rf = np.exp(r * self.dt)
        self.CALL = CALL
        
        # Set random seed for reproducibility
        np.random.seed(2021)
        tf.random.set_seed(2021)
        
        # Simulation: create a Rough Bergomi model instance and simulate paths.
        # Note: Here, we pass T as T and use N as the number of steps.

        self.Sts, self.V = self.model.simulate_paths(n_paths=self.M, n_steps=self.N, T=self.T, S0=self.S0)
        self.price = self.model.price_european(self.Sts, self.K_strike, self.T, self.r)
        
        # Store simulation paths for later use.
        self.Sts2 = self.Sts.copy()
        
        # Create time grid based on simulation output dimensions (should be N+1 points)
        self.t = np.linspace(0, self.N, self.V.shape[1])
        
    def plot_paths(self):
        plt.figure(figsize=(15,10))
        plt.subplot(2,1,1)
        plt.plot(self.t, self.Sts.T, alpha=0.1, color='blue')
        plt.title('Underlying Price Paths')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.grid(True)
        plt.subplot(2,1,2)
        plt.plot(self.t, self.V.T, alpha=0.1, color='blue')
        plt.title('Variance Process Paths')
        plt.xlabel('Time')
        plt.ylabel('Variance')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def prepare_data(self):
        # True sample paths for evaluation using log-moneyness
        X_true = np.log(self.Sts2[:, :-1] / self.K_strike)  # shape: (M, N)
        X_true = X_true.reshape((-1, self.N, 1))
        y_true = self.Sts.reshape((-1, self.N + 1, 1))
        self.X_true = X_true
        self.y_true = y_true

        # Generate additional training data (using the same simulation paths in this example)
        X = np.log(self.Sts2 / self.K_strike)        # shape: (M, N+1)
        n_feats = 1
        X2 = X[:, :-1].reshape((-1, self.N, n_feats))
        y2 = self.Sts2.reshape((-1, self.N + 1, n_feats))
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X2, y2, test_size=0.2, random_state=42)
        
    def deep_hedger(self, T_seq, n_feats):
        input_layer = Input(shape=(None, n_feats))
        
        # 1st GRU layer
        x = GRU(64, activation='tanh',
                return_sequences=True,
                kernel_initializer=initializers.RandomNormal(0, 0.1),
                bias_initializer=initializers.RandomNormal(0, 0.1))(input_layer)
        
        # Optional Dropout or BatchNorm
        x = Dropout(0.2)(x)
        
        # 2nd GRU layer
        x = GRU(32, activation='tanh',
                return_sequences=True,
                kernel_initializer=initializers.RandomNormal(0, 0.1),
                bias_initializer=initializers.RandomNormal(0, 0.1))(x)

        # Optional Dense layer(s)
        x = Dense(16, activation='relu')(x)
        x = Dropout(0.1)(x)

        # Output layer (delta hedge at each timestep)
        output_layer = Dense(1, activation='linear',
                            kernel_initializer=initializers.RandomNormal(),
                            bias_initializer=initializers.RandomNormal(0, 0.1))(x)
        
        model = Model(input_layer, output_layer)
        return model
    
    def MSHE_Loss(self, init_price, strike, T_seq):
        def lossFunction(y_true, y_pred):
            # For simplicity, ignore rf (assume rf=1)
            price_changes = tf.experimental.numpy.diff(y_true, n=1, axis=1)
            val = tf.reduce_sum(tf.math.multiply(price_changes, y_pred), axis=1)
            option_val = tf.math.maximum(y_true[:, -1] - strike, 0)
            return tf.math.reduce_mean(tf.math.square(-option_val + val + init_price))
        return lossFunction
    
    def cvarLoss(self, init_price, strike, T_seq, batch_size, proportion=0.01):
        num = int(batch_size * proportion)
        def lossFunction(y_true, y_pred):
            price_changes = tf.experimental.numpy.diff(y_true, n=1, axis=1)
            val = tf.reduce_sum(tf.math.multiply(price_changes, y_pred), axis=1)
            option_val = tf.math.maximum(y_true[:,-1,:] - strike, 0)
            error = tf.reshape(-(-option_val + val + init_price), [-1])
            CVaR, idx = tf.math.top_k(error, tf.constant(num, dtype=tf.int32))
            return tf.math.reduce_mean(CVaR)
        return lossFunction
    
    def build_and_compile_model(self, lr=0.005, loss_type='mshe'):
        self.n_feats = 1  # Using only log-moneyness
        # Build model with sequence length = N (i.e., T_seq = self.N)
        self.model = self.deep_hedger(self.N, self.n_feats)
        self.model.summary()
        print("Check Model", self.model.predict(np.zeros((1, self.N, 1))).reshape(-1))
        if loss_type == 'mshe':
            loss_func = self.MSHE_Loss(init_price=self.price, strike=self.K_strike, T_seq=self.N)
        else:
            # Uncomment below to use CVaR loss instead
            loss_func = self.cvarLoss(init_price=self.price, strike=self.K_strike, T_seq=self.N, batch_size=256, proportion=0.01)
            #loss_func = self.MSHE_Loss(init_price=self.price, strike=self.K_strike, T_seq=self.N)
        self.model.compile(optimizer=Adam(learning_rate=lr), loss=loss_func)
        
    def train_model(self, BATCH_SIZE=256, EPOCHS=50):
        logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        self.model.fit(self.X_train, self.y_train, epochs=EPOCHS, verbose=1, 
                       batch_size=BATCH_SIZE, callbacks=[tensorboard_callback, early_stopping_callback],
                       validation_data=(self.X_val, self.y_val), shuffle=False)
        
    def evaluate_model(self):
        # Pre-calculate deep hedge deltas on the true sample paths (X_true)
        N_SAMPLES = self.X_true.shape[0]
        deep_hedge_deltas = np.zeros((N_SAMPLES, self.N))
        for i in range(self.N):
            temp = self.model.predict(self.X_true[:, :i+1, :], batch_size=512)
            deep_hedge_deltas[:, i] = temp.reshape(-1, i+1)[:, i]
        
        # Calculate portfolio evolution
        deep_vals = np.zeros((N_SAMPLES, self.N + 1))
        deep_vals[:, 0] = self.price
        for t in range(1, self.N + 1):
            deep_vals[:, t] = self.rf * deep_vals[:, t - 1] + deep_hedge_deltas[:, t - 1] * (self.Sts[:, t] - self.rf * self.Sts[:, t - 1])
        
        deep_terminal_error = deep_vals[:, self.N] - np.maximum(self.Sts[:, self.N] - self.K_strike, 0)
        print("Mean hedging error:", np.mean(deep_terminal_error))
        print("Std of hedging error:", np.std(deep_terminal_error))
        self.deep_terminal_error = deep_terminal_error
        return deep_terminal_error
    
    def plot_error(self):
        plt.figure(figsize=(6,5))
        sns.histplot(self.deep_terminal_error).set_title("Deep-Hedger Error")
        plt.xlabel("Hedging Error")
        plt.ylabel("Frequency")
        plt.show()