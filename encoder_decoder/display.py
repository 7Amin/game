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
        image_number = random.randint(0, total_data - 1)
        image_number_1 = random.randint(0, total_data - 1)
        # np.random.seed(42)
        y_pred = self.model.predict(self.dataset)

        fig = plt.figure(figsize=(10, 7))

        fig.add_subplot(2, 2, 1)
        plt.imshow(self.dataset[image_number])
        plt.axis('off')
        plt.title("real")

        fig.add_subplot(2, 2, 2)
        plt.imshow(y_pred[image_number])
        plt.axis('off')
        plt.title("generated")

        fig.add_subplot(2, 2, 3)
        plt.imshow(self.dataset[image_number_1])
        plt.axis('off')
        plt.title("real_1")

        fig.add_subplot(2, 2, 4)
        plt.imshow(y_pred[image_number_1])
        plt.axis('off')
        plt.title("generated_1")

        plt.show()
