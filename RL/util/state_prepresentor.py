import numpy as np
import cv2


color = np.array([210, 164, 74]).mean()


def preprocess_state(state):
    # cv2.imwrite("state.png", state)
    # crop and resize the image
    image = state[29:188:2, ::2]

    # convert the image to greyscale
    image = image.mean(axis=2)

    # improve image contrast
    image[image == color] = 0

    # normalize the image
    # cv2.imwrite("state1.png", np.resize(image, (630, 480)))

    down_width = 128
    down_height = 128
    down_points = (down_width, down_height)
    image = cv2.resize(image[0], down_points, interpolation=cv2.INTER_LINEAR)
    image = image.reshape(down_width, down_width, 1)
    image = (image - 128.0) / 128.0

    # reshape the image
    image = np.expand_dims(image, axis=0)

    return image
