import multiprocessing
import re
import keras
import numpy as np
from keras import Model, Sequential
from keras.layers import Dense, Activation, Input
import tensorflow as tf
import sqlite3
import bitstring
from datetime import datetime
import os, os.path
from engine.engine.BitBoard import generate_bit_board
import gc


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
gc.collect()
tf.keras.backend.clear_session()
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=3072)])
            tf.config.experimental.set_memory_growth(gpu,True)
        except:
            print("GPU CONFIG ALREADY SET   ")
            pass

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

class DataGenerator(keras.utils.Sequence):
    def __init__(self, database, table_name, batch_size, max_rows):
        self.table_name = table_name
        self.batch_size = batch_size
        self.database = database
        self.max_rows = max_rows

    def __len__(self):
        if not self.max_rows:
            con = sqlite3.connect(self.database)
            cur = con.cursor()
            val = cur.execute(f"select count(*) from {self.table_name} limit {self.max_rows}").fetchone()[0]
            return int(val) // self.batch_size
        else:
            return int(self.max_rows) // self.batch_size

    def __getitem__(self, index):
        con = sqlite3.connect(self.database)
        cur = con.cursor()
        start = index*self.batch_size
        # print(f"OVO JE INDEX {index} OVO JE START {start} OVO JE SIZE {self.batch_size}")
        r = cur.execute(f"select * from {self.table_name} limit {self.batch_size} offset {start}")
        train_x = []
        train_y = []
        for row in r:
            x = row[1]
            x = ChessEngine.string_to_bits(x)
            if len(str(row[2])) == 0:
                continue
            if type(row[2]) == str and row[2].startswith('#'):
                y = row[2].replace('#', '')
            else:
                y = row[2]
            y = float(y)
            y = ChessEngine.clamp_eval(y)
            train_x.append(x)
            train_y.append(y)

        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y, dtype=np.float)
        return train_x, train_y

class EpochSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 5 == 0:
            save_dir = "./engine/engine/epoch_models"
            version = len([name for name in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, name))])
            save_path = save_dir + "/engine_v" + str(version) + ".h5"
            self.model.save(save_path)


class ChessEngine:
    def __init__(self):
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs/")
        self.chess_model = Sequential()
        self.chess_model.add(Input(shape=(789)))
        self.chess_model.add(Dense(1000, activation="relu"))
        self.chess_model.add(Dense(1000, activation="relu"))
        self.chess_model.add(Dense(1000, activation="relu"))
        self.chess_model.add(Dense(1000, activation="relu"))
        self.chess_model.add(Dense(1000, activation="relu"))
        self.chess_model.add(Dense(1000, activation="relu"))
        self.chess_model.add(Dense(1000, activation="relu"))
        self.chess_model.add(Dense(1000, activation="relu"))
        self.chess_model.add(Dense(1000, activation="relu"))
        self.chess_model.add(Dense(1))
        opt = keras.optimizers.Adam(learning_rate=0.002)
        self.chess_model.compile(loss=self.loss_fn, optimizer=opt)

    @staticmethod
    def string_to_bits(s):
        x = np.fromstring(s, 'u1') - ord('0')
        return x

    def train(self, max_rows=1000, epochs=5, batch_size=500):
        batch_size = 500
        data_generator = DataGenerator("./engine/engine/example.db", "chess_table", batch_size, max_rows)
        num_samples = len(data_generator)
        # dataset = tf.data.experimental.SqlDataset("sqlite",  "./engine/engine/example.db",
        #                                          "select (bitboard, eval) from chess_table", (tf.string, tf.string) )


        # def form(x, y):
        #     print(f" OVO JE x {x}       OVO JE y {y}")
        #     x = ChessEngine.string_to_bits(x)
        #     y = ChessEngine.clamp_eval(float(y))
        #     return (x, y)
        # dataset = dataset.map(form)

        print("ALLLL GOOOOOD")
        self.chess_model.fit(
            data_generator,
            epochs=epochs,
            verbose=1,
            steps_per_epoch=(num_samples),
            callbacks=[self.tensorboard_callback, EpochSaver()],
        )
        tf.keras.backend.clear_session()

    def loss_fn(self, y_true, y_pred):
        return (y_true - y_pred) ** 2

    def predict(self, bitboard):
        bitboard = self.string_to_bits(bitboard)
        bb = np.asarray([bitboard])
        return self.chess_model.predict(bb)

    @staticmethod
    def clamp_eval(val):
        if val <= -15:
            return -15
        elif val >= 15:
            return 15
        return val

    def save(self, save_dir = "./engine/engine/models"):
        version = len([name for name in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, name))])
        save_path = save_dir + "/engine_v" + str(version) + ".h5"
        self.chess_model.save_weights(save_path)

    def evaluate_fen(self, fen):
        bit_board = generate_bit_board(fen)
        return self.predict(bit_board)

    def load(self, path="./engine/engine/models/engine_v0.h5"):
        self.chess_model.load_weights(path)
        self.chess_model.summary()


def main():
    # engine = ChessEngine()
    # engine.train(max_rows=5, epochs=1)
    # engine.save()

    print("Num GPUs Available: ", tf.config.list_physical_devices("GPU"))
    pass


if __name__ == '__main__':
    main()
