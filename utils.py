import numpy as np
import os
from PIL import Image
from PIL import ImageOps
# from skimage import io
# from skimage import transform
# from skimage import util


def generate_random_strings(batch_size, seq_length, vector_dim):
    return np.random.randint(0, 2, size=[batch_size, seq_length, vector_dim]).astype(np.float32)


def one_hot_encode(x, dim):
    res = np.zeros(np.shape(x) + (dim, ), dtype=np.float32)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        res[it.multi_index][it[0]] = 1
        it.iternext()
    return res


def one_hot_decode(x):
    return np.argmax(x, axis=-1)


def five_hot_decode(x):
    x = np.reshape(x, newshape=np.shape(x)[:-1] + (5, 5))
    def f(a):
        return sum([a[i] * 5 ** i for i in range(5)])
    return np.apply_along_axis(f, -1, np.argmax(x, axis=-1))


def baseN(num,b):
    return ((num == 0) and  "0" ) or ( baseN(num // b, b).lstrip("0") + "0123456789abcdefghijklmnopqrstuvwxyz"[num % b])


class OmniglotDataLoader:
    def __init__(self, data_dir='./data', image_size=(20, 20), n_train_classses=1200, n_test_classes=423):
        self.data = []
        self.image_size = image_size
        for dirname, subdirname, filelist in os.walk(data_dir):
            if filelist:
                self.data.append(
                    # [np.reshape(
                    #     np.array(Image.open(dirname + '/' + filename).resize(image_size), dtype=np.float32),
                    #     newshape=(image_size[0] * image_size[1])
                    #     )
                    #     for filename in filelist]
                    # [io.imread(dirname + '/' + filename).astype(np.float32) / 255 for filename in filelist]
                    [Image.open(dirname + '/' + filename).copy() for filename in filelist]
                )

        self.train_data = self.data[:n_train_classses]
        self.test_data = self.data[-n_test_classes:]

    def fetch_batch(self, n_classes, batch_size, seq_length,
                    type='train',
                    sample_strategy='random',
                    augment=True,
                    label_type='one_hot'):
        if type == 'train':
            data = self.train_data
        elif type == 'test':
            data = self.test_data
        classes = [np.random.choice(range(len(data)), replace=False, size=n_classes) for _ in range(batch_size)]
        if sample_strategy == 'random':         # #(sample) per class may not be equal (sec 7)
            seq = np.random.randint(0, n_classes, [batch_size, seq_length])
        elif sample_strategy == 'uniform':      # #(sample) per class are equal
            seq = np.array([np.concatenate([[j] * int(seq_length / n_classes) for j in range(n_classes)])
                   for i in range(batch_size)])
            for i in range(batch_size):
                np.random.shuffle(seq[i, :])
        self.rand_rotate_init(n_classes, batch_size)
        seq_pic = [[self.augment(data[classes[i][j]][np.random.randint(0, len(data[classes[i][j]]))],
                                 batch=i, c=j,
                                 only_resize=not augment)
                   for j in seq[i, :]]
                   for i in range(batch_size)]
        if label_type == 'one_hot':
            seq_encoded = one_hot_encode(seq, n_classes)
            seq_encoded_shifted = np.concatenate(
                [np.zeros(shape=[batch_size, 1, n_classes]), seq_encoded[:, :-1, :]], axis=1
            )
        elif label_type == 'five_hot':
            label_dict = [[[int(j) for j in list(baseN(i, 5)) + [0] * (5 - len(baseN(i, 5)))]
                      for i in np.random.choice(range(5 ** 5), replace=False, size=n_classes)]
                     for _ in range(batch_size)]
            seq_encoded_ = np.array([[label_dict[b][i] for i in seq[b]] for b in range(batch_size)])
            seq_encoded = np.reshape(one_hot_encode(seq_encoded_, dim=5), newshape=[batch_size, seq_length, -1])
            seq_encoded_shifted = np.concatenate(
                [np.zeros(shape=[batch_size, 1, 25]), seq_encoded[:, :-1, :]], axis=1
            )
        return seq_pic, seq_encoded_shifted, seq_encoded

    def rand_rotate_init(self, n_classes, batch_size):
        self.rand_rotate_map = np.random.randint(0, 4, [batch_size, n_classes])

    def augment(self, image, batch, c, only_resize=False):
        if only_resize:
            image = ImageOps.invert(image.convert('L')).resize(self.image_size)
        else:
            rand_rotate = self.rand_rotate_map[batch, c] * 90                       # rotate by 0, pi/2, pi, 3pi/2
            image = ImageOps.invert(image.convert('L')) \
                .rotate(rand_rotate + np.random.rand() * 22.5 - 11.25,
                        translate=np.random.randint(-10, 11, size=2).tolist()) \
                .resize(self.image_size)   # rotate between -pi/16 to pi/16, translate bewteen -10 and 10
        np_image = np.reshape(np.array(image, dtype=np.float32),
                          newshape=(self.image_size[0] * self.image_size[1]))
        max_value = np.max(np_image)    # normalization is important
        if max_value > 0.:
            np_image = np_image / max_value
        return np_image
        # mat = transform.AffineTransform(translation=np.random.randint(-10, 11, size=2).tolist())
        # return np.reshape(
        #     util.invert(
        #         transform.resize(
        #             transform.warp(
        #                 transform.rotate(
        #                     util.invert(image),
        #                     angle=rand_rotate + np.random.rand() * 22.5 - 11.25
        #                 ), mat
        #             ), output_shape=self.image_size
        #         )
        #     ), newshape=(self.image_size[0] * self.image_size[1])
        # )
