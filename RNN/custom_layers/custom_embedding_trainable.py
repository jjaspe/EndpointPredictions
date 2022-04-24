
from keras.layers import Dense, Input
from keras import Sequential
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt

class CustomEmbeddingTrainable(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, vocab_input_size):
        super(CustomEmbeddingTrainable, self).__init__()
        self.sequential = Sequential([
            Dense(units=embedding_dim, activation='linear'),
            # Dense(units=vocab_input_size, activation='softmax')
        ])
        self.embedding_dim = embedding_dim
        self.vocab_input_size = vocab_input_size

    def call(self, x):
        rx = np.reshape(x, (x.shape[0], x.shape[1], 1))
        rx = self.sequential(rx)
        return rx

    
