import tensorflow as tf
from tensorflow.keras import backend as K


class AttentionBlock(tf.keras.Model):
    def __init__(self, num_heads=2, head_size=128, ff_dim=None, dropout=0, **kwargs):
        super().__init__(**kwargs)

        if ff_dim is None:
            ff_dim = head_size

        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)
        self.attention_dropout = tf.keras.layers.Dropout(dropout)
        self.attention_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.ff_conv1 = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')
        # self.ff_conv2 at build()
        self.ff_dropout = tf.keras.layers.Dropout(dropout)
        self.ff_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def build(self, input):
        input_shape = input[0]
        self.ff_conv2 = tf.keras.layers.Conv1D(filters=input_shape[-1], kernel_size=1)

    def call(self, inputs):
        x = self.attention(inputs[0], inputs[1])
        x = self.attention_dropout(x)
        x = self.attention_norm(inputs[1] + x)
        x = self.ff_conv1(x)
        x = self.ff_conv2(x)
        x = self.ff_dropout(x)
        x = self.ff_norm(inputs[1] + x)
        return x