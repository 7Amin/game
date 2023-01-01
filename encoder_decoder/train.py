import tensorflow as tf
import os
from keras.models import *
from keras.layers import *
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, fbeta_score
use_gpu = "no"

if use_gpu == "yes":
    print("-------------amin--------------------")
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("-------------amin--------------------")
    print(physical_devices)
    print("-------------amin--------------------")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import argparse
from model.model_factory import create_model
from dataset.dataset_factory import create_dataset
from display import PerformancePlotCallback
from config import train as train_config
import numpy as np
from config.predict import (
    HDF5_MODEL_PATH,
    LOG_DIR,
)

from tensorflow.python.keras.losses import mean_squared_error, mean_absolute_error
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-e", "--epochs", type=int, default=2500, metavar='>= 0', help="Training epochs")
parser.add_argument("-b", "--batch-size", type=int, default=16,
                    metavar='>= 0', help="Batch size")
parser.add_argument("-t", "--type", type=str, default="local", choices=["local", "OSC"], help="choose type")
parser.add_argument("-m", "--model", type=str, default="unet_vector", choices=["unet_low", "unet_vector", "unet"],
                    help="Number of model")
parser.add_argument("-d", "--dataset", type=int, default=0, choices=[0, 1, 2, 3], help="Number of dataset")
parser.add_argument("-dn", "--data_number", type=int, default=2, choices=[1, 2, 3], help="Number of data")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.01,
                    metavar='>= 0', help="Learning rate")
# parser.add_argument('-o', '--output', type=str, help='Output fname to save evaluation results to')
args = parser.parse_args()
np.random.seed(42)
tf.compat.v1.random.set_random_seed(42)

try:
    import shutil
    shutil.rmtree(LOG_DIR)
except Exception as e:
    print(e)
    pass

hdf5_model_path = HDF5_MODEL_PATH.format(args.model, args.dataset)

model = create_model(args.model)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
              loss=mean_squared_error,
              metrics=['mse', 'mae', 'mape'])

if True:
    print("************************Loading******************")
    model.load_weights(hdf5_model_path)
    print("************************Loaded********************")

# layer_name = 'conv2d_15'
# intermediate_layer_model = Model(inputs=model.input,
#                                  outputs=model.get_layer(layer_name).output)
# intermediate_output = intermediate_layer_model.predict(data)

data_input, data_input_test, data_output, data_output_test = create_dataset(args.dataset, args.data_number)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='mae',
        factor=0.9,
        patience=10,
        cooldown=1,
        min_lr=1e-6,
        verbose=1)

early_stop = EarlyStopping(
    monitor="val_mae",  # "val_loss",
    mode="min",
    verbose=1,
    patience=5,
    min_delta=0.0001,
    restore_best_weights=False)

checkpoint = ModelCheckpoint(
    hdf5_model_path,
    monitor='val_mae',  # or val_loss
    verbose=1,
    save_best_only=True,
    mode='min',
    # save_weights_only=True
)

tboard_callback = TensorBoard(
    log_dir=LOG_DIR,
    histogram_freq=0,
    write_graph=True,
    write_images=True)

display_data = PerformancePlotCallback(data_input_test)

history = model.fit(
    data_input,
    data_output,
    workers=10,
    epochs=args.epochs,
    shuffle=True,
    verbose=1,
    validation_data=(data_input_test, data_output_test),
    validation_freq=1,
    callbacks=[
        tboard_callback,
        early_stop,
        display_data,
        reduce_lr,
        checkpoint,
    ])
