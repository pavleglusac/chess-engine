import numpy as np
from keras import Model, Sequential
from keras.layers import Dense, Activation
import tensorflow as tf
import sqlite3
import bitstring
from datetime import datetime
import os, os.path


def pretty_print_bitboard(x):
    # print(x)
    i = 0
    j = 0
    for b in x:
        print(int(b), end="")
        i += 1
        if i == 8:
            i = 0
            j += 1
            print()
        if j == 8:
            j = 0
            print("\n--------")
    print()
    print(len(x))

class ChessEngine:
    def __init__(self):
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs/")
        self.chess_model = Sequential()
        self.chess_model.add(Dense(789, activation="relu"))
        self.chess_model.add(Dense(1000, activation="relu"))
        self.chess_model.add(Dense(1000, activation="relu"))
        self.chess_model.add(Dense(1000, activation="relu"))
        self.chess_model.add(Dense(1000, activation="relu"))
        self.chess_model.add(Dense(1000, activation="relu"))
        self.chess_model.add(Dense(1000, activation="relu"))
        self.chess_model.add(Dense(1000, activation="relu"))
        self.chess_model.add(Dense(1000, activation="relu"))
        self.chess_model.add(Dense(1))
        self.chess_model.compile(loss=self.loss_fn, optimizer="adam")

    def train(self, max_rows=10000, epochs=5):
        con = sqlite3.connect("example.db")
        cur = con.cursor()
        r = cur.execute("select * from chess_table limit " + str(max_rows))
        train_x = []
        train_y = []
        for row in r:
            x = row[1]
            x = np.fromstring(x, 'u1') - ord('0')
            # pretty_print_bitboard(x)
            if len(str(row[2])) == 0:
                continue

            if type(row[2]) == str and row[2].startswith('#'):
                y = row[2].replace('#', '')
            else:
                y = row[2]
            y = float(y)
            y = self.clamp_eval(y)
            train_x.append(x)
            train_y.append(y)

        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y, dtype=np.float)
        self.chess_model.fit(x=train_x,
                             y=train_y,
                             epochs=epochs,
                             callbacks=[self.tensorboard_callback],
                             batch_size=64)

    def loss_fn(self, y_true, y_pred):
        return (y_true - y_pred) ** 2

    def predict(self, input):
        x = np.frombuffer(input, dtype=np.uint8)
        x = np.unpackbits(x, axis=0).astype(np.single)
        return self.chess_model.predict(x)

    def clamp_eval(self, val):
        if val <= -15:
            return -15
        elif val >= 15:
            return 15
        return val

    def save(self):
        save_dir = "./models"
        version = len([name for name in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, name))])
        save_path = save_dir + "/engine_v" + str(version) + ".h5"
        self.chess_model.save(save_path)

    def evaluate_fen(self, fen):
        self.predict(fen)


def main():
    engine = ChessEngine()
    engine.train(max_rows=10000, epochs=10)
    # engine.save()
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


if __name__ == '__main__':
    main()
