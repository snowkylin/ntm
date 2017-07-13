import numpy as np
import os
from PIL import Image


def generate_random_strings(batch_size, seq_length, vector_dim):
    return np.random.randint(0, 2, size=[batch_size, seq_length, vector_dim]).astype(np.float32)


def one_hot(x, dim):
    res = np.zeros(np.shape(x) + (dim, ), dtype=np.float32)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        res[it.multi_index][it[0]] = 1
        it.iternext()
    return res


class OmniglotDataLoader:
    def __init__(self, data_dir='./data', image_size=(20, 20), n_train_classses=1200, n_test_classes=423):
        self.data = []
        for dirname, subdirname, filelist in os.walk(data_dir):
            if filelist:
                self.data.append(
                    [np.reshape(
                        np.array(Image.open(dirname + '/' + filename).resize(image_size), dtype=np.float32),
                        newshape=(image_size[0] * image_size[1])
                        )
                        for filename in filelist]
                )
        self.n_classes = len(self.data)
        self.train_data = self.data[:n_train_classses]
        self.test_data = self.data[-n_test_classes:]

    def fetch_batch(self, n_classes, batch_size, seq_length, type='train'):
        if type == 'train':
            data = self.train_data
        else:
            data = self.test_data
        seq = np.random.randint(0, n_classes, [batch_size, seq_length])
        classes = [np.random.choice(range(self.n_classes), replace=False, size=n_classes) for _ in range(batch_size)]
        seq_pic = [[data[classes[i][j]][np.random.randint(0, len(data[classes[i][j]]))]
                   for j in seq[i, :]]
                   for i in range(batch_size)]
        seq_one_hot = one_hot(seq, n_classes)
        seq_one_hot_shifted = np.concatenate(
            [np.zeros(shape=[batch_size, 1, n_classes]), seq_one_hot[:, 1:, :]], axis=1
        )
        return seq_pic, seq_one_hot_shifted, seq_one_hot