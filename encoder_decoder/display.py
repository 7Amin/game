import tensorflow as tf
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class PerformancePlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset):
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        total_data = len(self.dataset)
        image_number = random.randint(0, total_data - 1)
        np.random.seed(42)
        y_pred = self.model.predict(self.dataset)

        fig = plt.figure(figsize=(10, 7))

        fig.add_subplot(1, 2, 1)
        plt.imshow(self.dataset[image_number])
        plt.axis('off')
        plt.title("First")

        fig.add_subplot(1, 2, 2)
        plt.imshow(y_pred[image_number])
        plt.axis('off')
        plt.title("Second")

        plt.show()
