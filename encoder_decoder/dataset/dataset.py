from encoder_decoder.config import dataset as dataset_config
from sklearn.model_selection import train_test_split

import tensorflow as tf
import cv2
import numpy as np
import os


np.random.seed(42)


def get_dataset(data_number=1, base_url=dataset_config.PATH, image_size=(512, 512)):
    X = []
    base_url_images = base_url.format(data_number)
    arr_images = os.listdir(base_url_images)
    i = 0
    for image_name in arr_images:
        i = i + 1
        # if (i % 5) != 0:
        #     continue
        path = base_url_images + image_name
        image = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), image_size)
        X.append(image)
    X = np.array(X) / 255.0
    data_input, data_input_test, data_output, data_output_test = train_test_split(X, X, random_state=1, train_size=.90)

    return data_input, data_input_test, data_output, data_output_test
