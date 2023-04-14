import tensorflow as tf
tf.random.set_seed(123)


class FeatureExtraction(tf.keras.layers.Layer):
    def __init__(self):
        super(FeatureExtraction, self).__init__()

        self.model = tf.keras.applications.Xception(weights="weights/xception_weights_tf_dim_ordering_tf_kernels_notop.h5",
                                                    include_top=False, input_shape=(480, 640, 3),
                                                    pooling="avg")
        self.layer_name = 'block8_sepconv2_act'
        self.intermediate_layer_model = tf.keras.Model(inputs=self.model.input,
                                                       outputs=self.model.get_layer(self.layer_name).output)
        for layer in self.model.layers:
             layer.trainable = False

    def call(self, inputs):
        x = tf.keras.applications.xception.preprocess_input(inputs)
        x = self.intermediate_layer_model(x)
        # for layers in self.intermediate_layer_model.layers[1:12]:
        #     if layers.name != 'block8_sepconv2_act':
        #         x = layers(x)
        #     else:
        #         pass
        return x


class DownscaleBlock(tf.keras.layers.Layer):
    def __init__(
            self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.convA = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = tf.keras.layers.LeakyReLU()
        self.reluB = tf.keras.layers.LeakyReLU()
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.bn2b = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.MaxPool2D((2, 2), (2, 2))

    def call(self, input_tensor):
        d = self.convA(input_tensor)
        x = self.bn2a(d)
        x = self.reluA(x)

        x = self.convB(x)
        x = self.bn2b(x)
        x = self.reluB(x)

        x += d
        p = self.pool(x)
        return x, p


class UpscaleBlock(tf.keras.layers.Layer):
    def __init__(
        self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.us = tf.keras.layers.UpSampling2D((2, 2))
        self.convA = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.reluB = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.bn2b = tf.keras.layers.BatchNormalization()
        self.conc = tf.keras.layers.Concatenate()

    def call(self, x, skip):
        x = self.us(x)
        concat = self.conc([x, skip])
        x = self.convA(concat)
        x = self.bn2a(x)
        x = self.reluA(x)

        x = self.convB(x)
        x = self.bn2b(x)
        x = self.reluB(x)

        return x


class BottleNeckBlock(tf.keras.layers.Layer):
    def __init__(
            self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.convA = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = tf.keras.layers.LeakyReLU()
        self.reluB = tf.keras.layers.LeakyReLU()

    def call(self, x):
        x = self.convA(x)
        x = self.reluA(x)
        x = self.convB(x)
        x = self.reluB(x)
        return x
