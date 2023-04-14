import tensorflow as tf
from Model.model import DepthEstimationModel
from utils.dataset import DataGenerator
import yaml
from keras.callbacks import TensorBoard
import argparse
import os
import json


class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        if tf.math.is_nan(logs.get('loss')):
            print("   Got Nan")
            self.model.stop_training = True

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


parser = argparse.ArgumentParser()
parser.add_argument('--loadconfig', type=dir_path)
args = parser.parse_args()
with open('config/config.yaml', 'r') as file:
    load_config = yaml.safe_load(file)

model_config = load_config["Model"]
learning_rate_schedule_config = model_config["learning rate schedule"]
early_stopping_config = model_config["early stopping"]
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate_schedule_config['learning_rate'],
            decay_steps=learning_rate_schedule_config['learning_rate_decay_steps'],
            decay_rate=learning_rate_schedule_config['learning_rate_decay_rate'])

tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule,
    amsgrad=False,
    clipnorm=1.0
)

model = DepthEstimationModel()
# Define the loss function
callback = tf.keras.callbacks.EarlyStopping(monitor=early_stopping_config["type"],
                                            patience=early_stopping_config["patience"])

# Compile the model
model.compile(optimizer)
train = DataGenerator(model_config["dataset path"], "indoors", "train", "config/images.json",
                      batch_size=model_config["batch size"], dim=(480, 640))
model.build((None, 480, 640, 3))
checkpoint_path = "/home/zahra/Hesam/depth estimation/ckpnt/ckpint.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
print(model.summary())
history = model.fit(
    train,
    epochs=model_config["epoch"],
    callbacks=[callback, tensorboard_callback, LossHistory(), cp_callback])
model.save("weights/model_unet")
model.save_weights("weights/weights_unet.h5")
with open("weights/history1.json", 'a') as f:
    json.dump(history.history, f)

