import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GRU, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


class DeepHedgerTF:
    def __init__(self, model, S0=100, K_strike=100, M=10**5, N=100, T=30, r=0.0, CALL=True):
        """
        Initialize Deep Hedging framework with a stochastic model.
        """
        self.model = model
        self.S0 = S0
        self.K_strike = K_strike
        self.M = M
        self.N = N
        self.T = T
        self.dt = T / N
        self.r = r
        self.rf = np.exp(r * self.dt)
        self.CALL = CALL

        # Set random seed
        np.random.seed(2021)
        tf.random.set_seed(2021)

        # Simulate asset paths and compute option price
        self.S, self.V = self.model.simulate_paths(n_paths=self.M, n_steps=self.N, T=self.T, S0=self.S0)
        self.price = self.model.price_european(self.S, self.K_strike, self.T, self.r)
        self.t = np.linspace(0, self.N, self.V.shape[1])

    def plot_paths(self):
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        plt.plot(self.t, self.S.T, alpha=0.1, color='blue')
        plt.title('Underlying Price Paths')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(self.t, self.V.T, alpha=0.1, color='blue')
        plt.title('Variance Process Paths')
        plt.xlabel('Time')
        plt.ylabel('Variance')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def prepare_data(self):
        X_true = np.log(self.S[:, :-1] / self.K_strike).reshape((-1, self.N, 1))
        y_true = self.S.reshape((-1, self.N + 1, 1))
        self.X_true = X_true
        self.y_true = y_true

        X = np.log(self.S / self.K_strike)
        self.n_feats = 1
        X2 = X[:, :-1].reshape((-1, self.N, self.n_feats))
        y2 = self.S.reshape((-1, self.N + 1, self.n_feats))
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X2, y2, test_size=0.2, random_state=42)

    def deep_hedger(self, T_seq, n_feats, gru_layers=2, hidden_size=64):
        inputs = Input(shape=(None, n_feats))
        x = inputs

        for _ in range(gru_layers):
            x = GRU(hidden_size, activation='tanh', return_sequences=True)(x)
            x = Dropout(0.2)(x)

        x = Dense(hidden_size // 2, activation='relu')(x)
        outputs = Dense(1, activation='linear')(x)
        return Model(inputs, outputs)

    def MSHE_Loss(self, init_price, strike):
        def loss_fn(y_true, y_pred):
            price_changes = tf.experimental.numpy.diff(y_true, axis=1)
            hedge_value = tf.reduce_sum(price_changes * y_pred, axis=1)
            option_value = tf.maximum(y_true[:, -1] - strike, 0)
            return tf.reduce_mean(tf.square(-option_value + hedge_value + init_price))
        return loss_fn

    def cvarLoss(self, init_price, strike, batch_size, proportion=0.01):
        num = int(batch_size * proportion)

        def loss_fn(y_true, y_pred):
            price_changes = tf.experimental.numpy.diff(y_true, axis=1)
            hedge_value = tf.reduce_sum(price_changes * y_pred, axis=1)
            option_value = tf.maximum(y_true[:, -1, 0] - strike, 0)
            error = -(-option_value + hedge_value + init_price)
            cvar, _ = tf.math.top_k(error, num)
            return tf.reduce_mean(cvar)

        return loss_fn

    def entropicLoss(self, init_price, strike, gamma=1.0):
        def loss_fn(y_true, y_pred):
            price_changes = tf.experimental.numpy.diff(y_true, axis=1)
            hedge_value = tf.reduce_sum(price_changes * y_pred, axis=1)
            option_value = tf.maximum(y_true[:, -1] - strike, 0)
            pnl = -option_value + hedge_value + init_price
            max_pnl = tf.reduce_max(-gamma * pnl)
            stabilized = tf.exp(-gamma * pnl - max_pnl)
            mean_stabilized = tf.reduce_mean(stabilized)
            return (1.0 / gamma) * (tf.math.log(mean_stabilized) + max_pnl)
        return loss_fn

    def build_and_compile_model(self, lr=0.005, loss_type='cvar', gru_layers=2, hidden_size=64, optimizer='adam', gamma=1.0):
        self.model = self.deep_hedger(self.N, self.n_feats, gru_layers, hidden_size)

        optimizers = {
            'adam': Adam(learning_rate=lr),
            'rmsprop': tf.keras.optimizers.RMSprop(learning_rate=lr),
            'sgd': tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        }
        opt = optimizers.get(optimizer)
        if opt is None:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        loss_funcs = {
            'cvar': self.cvarLoss(self.price, self.K_strike, batch_size=256),
            'mshe': self.MSHE_Loss(self.price, self.K_strike),
            'entropic': self.entropicLoss(self.price, self.K_strike, gamma=gamma)
        }
        loss_func = loss_funcs.get(loss_type)
        if loss_func is None:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        self.model.compile(optimizer=opt, loss=loss_func)

    def train_model(self, batch_size=256, epochs=50):
        logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        callbacks = [
            tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        ]
        self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val, self.y_val),
            callbacks=callbacks,
            shuffle=False,
            verbose=0
        )

    def predict(self, X_input):
        if not hasattr(self, 'model'):
            raise ValueError("Model not built. Please call `build_and_compile_model()` first.")
        return self.model.predict(X_input)

    def evaluate_model(self):
        n_samples = self.X_true.shape[0]
        deltas = np.zeros((n_samples, self.N))

        for i in range(self.N):
            pred = self.model.predict(self.X_true[:, :i+1, :], batch_size=512, verbose=0)
            deltas[:, i] = pred[:, -1, 0]

        vals = np.zeros((n_samples, self.N + 1))
        vals[:, 0] = self.price

        for t in range(1, self.N + 1):
            dS = self.S[:, t] - self.rf * self.S[:, t - 1]
            vals[:, t] = self.rf * vals[:, t - 1] + deltas[:, t - 1] * dS

        err = vals[:, -1] - np.maximum(self.S[:, -1] - self.K_strike, 0)
        print("Mean hedging error:", np.mean(err))
        print("Std of hedging error:", np.std(err))
        self.deep_terminal_error = err
        return err

    def plot_error(self):
        if hasattr(self, 'deep_terminal_error'):
            plt.figure(figsize=(6, 5))
            sns.histplot(self.deep_terminal_error, bins=50)
            plt.title("Deep-Hedger Hedging Error")
            plt.xlabel("Hedging Error")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.show()

    def plot_pnl(self):
        if hasattr(self, 'deep_terminal_pnl'):
            plt.figure(figsize=(6, 5))
            sns.histplot(self.deep_terminal_pnl, bins=50)
            plt.title("Deep-Hedger PnL")
            plt.xlabel("PnL")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.show()