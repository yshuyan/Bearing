import os
import numpy as np


def mkdir(path):

    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)


def get_random_block_from_test(data, size):
    sample = np.random.randint(len(data), size=size)
    data = data[sample]
    return data
