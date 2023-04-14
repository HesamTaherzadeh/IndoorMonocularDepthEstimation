import tensorflow as tf
from Model.layers import FeatureExtraction, UpscaleBlock, BottleNeckBlock, DownscaleBlock
from tensorflow import keras
import numpy as np
import keras.backend as K


class DepthEstimationModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.ssim_loss_weight = 0.85
        self.l1_loss_weight = 0.1
        self.edge_loss_weight = 0.9
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        f = [16, 32, 64, 128]
        # self.fe = FeatureExtraction()
        self.downscale_blocks = [
            DownscaleBlock(f[0], name="db0"),
            DownscaleBlock(f[1], name="db1"),
            DownscaleBlock(f[2], name="db2"),
            DownscaleBlock(f[3], name="db3"),
        ]
        self.bottle_neck_block = BottleNeckBlock(256)
        self.upscale_blocks = [
            UpscaleBlock(f[3], name="ub1"),
            UpscaleBlock(f[2], name="ub2"),
            UpscaleBlock(f[1], name="ub3"),
            UpscaleBlock(f[0], name="ub4"),
        ]
        self.conv_layer = tf.keras.layers.Conv2D(1, (1, 1), padding="valid", activation='sigmoid',
                                                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4))
        self.droputs = [
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dropout(0.2)
        ]
        self.bn = [
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.BatchNormalization()
        ]

    def calculate_loss(self, y_true, y_pred):
        # Invert depth maps and clip values to prevent NaNs
        theta = 0.1
        maxDepthVal = 1.0
        l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

        # Edges
        dy_true, dx_true = tf.image.image_gradients(y_true)
        dy_pred, dx_pred = tf.image.image_gradients(y_pred)
        l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

        # Structural similarity (SSIM) index
        l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

        # Weights
        w1 = 1.0
        w2 = 1.0
        w3 = theta
        # tf.print("    ")
        # tf.print(tf.reduce_any(tf.math.is_nan(y_true)))
        # tf.print(tf.reduce_any(tf.math.is_nan(y_pred)))
        # tf.print(l_ssim)
        # tf.print(K.mean(l_edges))
        # tf.print(K.mean(l_depth))
        return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))

    @property
    def metrics(self):
        return [self.loss_metric]

    def train_step(self, data):
        inp, target = data
        inp = tf.image.convert_image_dtype(inp, tf.float32)
        target = tf.image.convert_image_dtype(target, tf.float32)
        with tf.GradientTape() as tape:
            pred = self(inp)
            loss = self.calculate_loss(target, pred)
            loss = tf.convert_to_tensor(loss)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_metric.update_state(loss)
        return {
            "loss": self.loss_metric.result(),
        }

    def test_step(self, batch_data):
        input, target = batch_data
        pred = self(input, training=False)
        loss = self.calculate_loss(target, pred)

        self.loss_metric.update_state(loss)
        return {
            "loss": self.loss_metric.result(),
        }

    def call(self, x):
        c1, p1 = self.downscale_blocks[0](x)
        c2, p2 = self.downscale_blocks[1](p1)
        p2 = self.bn[0](p2)
        p2 = self.droputs[0](p2)
        c3, p3 = self.downscale_blocks[2](p2)
        c4, p4 = self.downscale_blocks[3](p3)
        p4 = self.bn[1](p4)
        p4 = self.droputs[1](p4)
        bn = self.bottle_neck_block(p4)

        u1 = self.upscale_blocks[0](bn, c4)
        u2 = self.upscale_blocks[1](u1, c3)
        u2 = self.bn[2](u2)
        u2 = self.droputs[2](u2)
        u3 = self.upscale_blocks[2](u2, c2)
        u4 = self.upscale_blocks[3](u3, c1)
        u4 = self.bn[3](u4)
        u4 = self.droputs[3](u4)

        x = self.conv_layer(u4)
        return x
