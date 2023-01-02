import numpy as np

color = np.array([210, 164, 74]).mean()


def preprocess_state(state):
    # cv2.imwrite("state.png", state)
    # crop and resize the image
    image = state[9:168:2, ::2]

    # convert the image to greyscale
    image = image.mean(axis=2)

    # improve image contrast
    image[image == color] = 0

    # normalize the image
    # cv2.imwrite("state1.png", np.resize(image, (630, 480)))

    image = image.reshape(80, 80, 1)
    image = (image - 128) / 128 - 1

    # reshape the image
    image = np.expand_dims(image, axis=0)

    return image
