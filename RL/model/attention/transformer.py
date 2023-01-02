import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from model.basic_model import BasicModel
import random


class PatchExtract(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(PatchExtract, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        print(images.shape)
        patches = tf.image.extract_patches(
            images=images,
            sizes=(1, self.patch_size, self.patch_size, 1),
            strides=(1, self.patch_size, self.patch_size, 1),
            rates=(1, 1, 1, 1),
            padding="VALID",
        )
        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1]
        return tf.reshape(patches, (batch_size, patch_num * patch_num, patch_dim))


class PatchEmbedding(layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        return self.proj(patch) + self.pos_embed(pos)


def external_attention(
    x, dim, num_heads, dim_coefficient=4, attention_dropout=0, projection_dropout=0
):
    _, num_patch, channel = x.shape
    assert dim % num_heads == 0
    num_heads = num_heads * dim_coefficient

    x = layers.Dense(dim * dim_coefficient)(x)
    # create tensor [batch_size, num_patches, num_heads, dim*dim_coefficient//num_heads]
    x = tf.reshape(
        x, shape=(-1, num_patch, num_heads, dim * dim_coefficient // num_heads)
    )
    x = tf.transpose(x, perm=[0, 2, 1, 3])
    # a linear layer M_k
    attn = layers.Dense(dim // dim_coefficient)(x)
    # normalize attention map
    attn = layers.Softmax(axis=2)(attn)
    # dobule-normalization
    attn = attn / (1e-9 + tf.reduce_sum(attn, axis=-1, keepdims=True))
    attn = layers.Dropout(attention_dropout)(attn)
    # a linear layer M_v
    x = layers.Dense(dim * dim_coefficient // num_heads)(attn)
    x = tf.transpose(x, perm=[0, 2, 1, 3])
    x = tf.reshape(x, [-1, num_patch, dim * dim_coefficient])
    # a linear layer to project original dim
    x = layers.Dense(dim)(x)
    x = layers.Dropout(projection_dropout)(x)
    return x


def mlp(x, embedding_dim, mlp_dim, drop_rate=0.2):
    x = layers.Dense(mlp_dim, activation=tf.nn.gelu)(x)
    x = layers.Dropout(drop_rate)(x)
    x = layers.Dense(embedding_dim)(x)
    x = layers.Dropout(drop_rate)(x)
    return x


def transformer_encoder(
    x,
    embedding_dim,
    mlp_dim,
    num_heads,
    dim_coefficient,
    attention_dropout,
    projection_dropout,
    attention_type="external_attention",
):
    residual_1 = x
    x = layers.BatchNormalization(epsilon=1e-5)(x)
    if attention_type == "external_attention":
        x = external_attention(
            x,
            embedding_dim,
            num_heads,
            dim_coefficient,
            attention_dropout,
            projection_dropout,
        )
    elif attention_type == "self_attention":
        x = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dim, dropout=attention_dropout
        )(x, x)
    x = layers.add([x, residual_1])
    residual_2 = x
    x = layers.BatchNormalization(epsilon=1e-5)(x)
    x = mlp(x, embedding_dim, mlp_dim)
    x = layers.add([x, residual_2])
    return x


class Transformer(BasicModel):
    def __init__(self, state_size, action_size, update_rate):
        super().__init__(state_size, action_size, update_rate)
        self.name = "Transformer"
        self.main_network = self.build_network()
        self.target_network = self.build_network()
        self.target_network.set_weights(self.main_network.get_weights())

        self.patch_size = 2  # Size of the patches to be extracted from the input images.
        self.num_patches = (state_size[0] // self.patch_size) ** 2  # Number of patch
        self.attention_type = "self_attention"
        self.embedding_dim = 64  # Number of hidden units.
        self.mlp_dim = 64
        self.num_heads = 4
        self.dim_coefficient = 4
        self.attention_dropout = 0.2
        self.projection_dropout = 0.2
        self.num_transformer_blocks = 8

    def build_network(self):
        inputs = layers.Input(shape=self.state_size)
        # Image augment
        # x = data_augmentation(inputs)
        # Extract patches.
        x = PatchExtract(self.patch_size)(inputs)
        # Create patch embedding.
        x = PatchEmbedding(self.num_patches, self.embedding_dim)(x)
        # Create Transformer block.
        for _ in range(self.num_transformer_blocks):
            x = transformer_encoder(
                x,
                self.embedding_dim,
                self.mlp_dim,
                self.num_heads,
                self.dim_coefficient,
                self.attention_dropout,
                self.projection_dropout,
                self.attention_type,
            )

        x = layers.GlobalAvgPool1D()(x)
        x = layers.Dense(512, activation='relu')(x)
        outputs = layers.Dense(self.action_size, activation="softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer=Adam())
        return model
