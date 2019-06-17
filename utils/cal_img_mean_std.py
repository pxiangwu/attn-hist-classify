import numpy as np
import os
import cv2
import logging
import math


root_data_path = '/media/pwu/Data/2D_data/pathology_images/mkfold/'
magnification = ['40X', '100X', '200X', '400X']


def cal_mean_std_per_mag():
    logging.basicConfig(filename="mean_std.txt", level=logging.INFO)

    for i in range(5):
        data_path = os.path.join(root_data_path, 'fold' + str(i + 1))
        train_data_path = os.path.join(data_path, 'train')
        test_data_path = os.path.join(data_path, 'test')

        for mag in magnification:
            img_dataset = []

            # read train images
            train_img_path = os.path.join(train_data_path, mag)
            train_imgs = [f for f in os.listdir(train_img_path) if os.path.isfile(os.path.join(train_img_path, f))]

            for img_file in train_imgs:
                img = cv2.imread(os.path.join(train_img_path, img_file))
                if img.shape != (460, 700, 3):
                    img = cv2.resize(img, (700, 460), interpolation=cv2.INTER_CUBIC)

                img_dataset.append(img)

            # read test images
            # test_img_path = os.path.join(test_data_path, mag)
            # test_imgs = [f for f in os.listdir(test_img_path) if os.path.isfile(os.path.join(test_img_path, f))]
            #
            # for img_file in test_imgs:
            #     img = cv2.imread(os.path.join(test_img_path, img_file))
            #     if img.shape != (460, 700, 3):
            #         img = cv2.resize(img, (700, 460), interpolation=cv2.INTER_CUBIC)
            #
            #     img_dataset.append(img)

            img_dataset = np.stack(img_dataset)
            mean, std = np.mean(img_dataset, axis=(0, 1, 2)), np.std(img_dataset, axis=(0, 1, 2))

            print('fold: {}, mag: {}\n mean {}, std {}'.format(i + 1, mag, mean, std))
            logging.info('fold: {}, mag: {}\n mean {}, std {}'.format(i + 1, mag, mean, std))


def cal_mean_std_across_mag():
    logging.basicConfig(filename="mean_std_across_mag.txt", level=logging.INFO)

    for i in range(1, 2):
        data_path = os.path.join(root_data_path, 'fold' + str(i + 1))
        train_data_path = os.path.join(data_path, 'train')

        running_mean_std_r = Welford()
        running_mean_std_g = Welford()
        running_mean_std_b = Welford()

        progress = 0
        for mag in magnification:
            # read train images
            train_img_path = os.path.join(train_data_path, mag)
            train_imgs = [f for f in os.listdir(train_img_path) if os.path.isfile(os.path.join(train_img_path, f))]

            for img_file in train_imgs:
                img = cv2.imread(os.path.join(train_img_path, img_file))

                img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)

                flatten_img = np.reshape(img, (-1, 3))

                running_mean_std_r(flatten_img[:, 0])
                running_mean_std_g(flatten_img[:, 1])
                running_mean_std_b(flatten_img[:, 2])

                progress += 1

                if progress % 5 == 0:
                    print(">>> progress {}".format(progress))

        mean = [running_mean_std_r.mean, running_mean_std_g.mean, running_mean_std_b.mean]
        std = [running_mean_std_r.std, running_mean_std_g.std, running_mean_std_b.std]

        print('fold: {} \n mean {}, std {}'.format(i + 1, mean, std))
        logging.info('fold: {} \n mean {}, std {}'.format(i + 1, mean, std))


# borrowed from https://gist.github.com/alexalemi/2151722
class Welford(object):
    """ Implements Welford's algorithm for computing a running mean
    and standard deviation as described at:
        http://www.johndcook.com/standard_deviation.html
    can take single values or iterables
    Properties:
        mean    - returns the mean
        std     - returns the std
        meanfull- returns the mean and std of the mean
    Usage:
        >>> foo = Welford()
        >>> foo(range(100))
        >>> foo
        <Welford: 49.5 +- 29.0114919759>
        >>> foo([1]*1000)
        >>> foo
        <Welford: 5.40909090909 +- 16.4437417146>
        >>> foo.mean
        5.409090909090906
        >>> foo.std
        16.44374171455467
        >>> foo.meanfull
        (5.409090909090906, 0.4957974674244838)
    """

    def __init__(self, lst=None):
        self.k = 0
        self.M = 0
        self.S = 0

        self.__call__(lst)

    def update(self, x):
        if x is None:
            return
        self.k += 1
        newM = self.M + (x - self.M) * 1. / self.k
        newS = self.S + (x - self.M) * (x - newM)
        self.M, self.S = newM, newS

    def consume(self, lst):
        lst = iter(lst)
        for x in lst:
            self.update(x)

    def __call__(self, x):
        if hasattr(x, "__iter__"):
            self.consume(x)
        else:
            self.update(x)

    @property
    def mean(self):
        return self.M

    @property
    def meanfull(self):
        return self.mean, self.std / math.sqrt(self.k)

    @property
    def std(self):
        if self.k == 1:
            return 0
        return math.sqrt(self.S / (self.k - 1))

    def __repr__(self):
        return "<Welford: {} +- {}>".format(self.mean, self.std)


if __name__ == '__main__':
    cal_mean_std_across_mag()
