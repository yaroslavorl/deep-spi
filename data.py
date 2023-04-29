import numpy as np
import cv2


def read_img(path):
    with open(path, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)
        img = np.reshape(everything, (-1, 3, 96, 96))

    return img


def resize_img(_imgs, size):
    train_set = np.zeros((size, 64, 64, 1))

    for i in range(size):
        img = cv2.cvtColor(_imgs[i, :, :, :], cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64, 64))
        img = np.expand_dims(img, axis=-1)
        train_set[i, :, :, :] = img

    return train_set


def get_dataframe(size_train, path):
    train_x = read_img(path)
    train_set = resize_img(np.transpose(train_x, (0, 3, 2, 1)), size=train_x.shape[0]).transpose((0, 3, 1, 2))

    train_x = train_set[:size_train].astype('float32') / 255.
    test_x = train_set[size_train:].astype('float32') / 255.

    return train_x, test_x
