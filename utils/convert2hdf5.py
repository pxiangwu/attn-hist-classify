import h5py
import os
from os import listdir
import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    data_path = '/media/pwu/Data/2D_data/pathology_images/mkfold'
    num_fold = 5

    magnification = ['40X', '100X', '200X', '400X']

    for i in range(num_fold):
        fold_path = os.path.join(data_path, 'fold' + str(i + 1))
        train_path = os.path.join(fold_path, 'train')
        test_path = os.path.join(fold_path, 'test')

        for mag in magnification:
            train_img_path = os.path.join(train_path, mag)
            test_img_path = os.path.join(test_path, mag)

            train_imgs = [f for f in listdir(train_img_path) if os.path.isfile(os.path.join(train_img_path, f))]
            test_imgs = [f for f in listdir(test_img_path) if os.path.isfile(os.path.join(test_img_path, f))]

            # read and save images
            train_imgs_dataset = []
            test_imgs_dataset = []

            train_label_dataset = []
            test_label_dataset = []

            for img_file in train_imgs:
                img = cv2.imread(os.path.join(train_img_path, img_file))
                if img.shape != (460, 700, 3):
                    img = cv2.resize(img, (700, 460), interpolation=cv2.INTER_CUBIC)
                    print('new image size', img.shape)

                train_imgs_dataset.append(img)

                tokens = img_file.split("_")
                if tokens[1] == 'B':
                    label = 0
                elif tokens[1] == 'M':
                    label = 1
                else:
                    raise ValueError('Unexpected Label!')
                train_label_dataset.append(label)

            for img_file in test_imgs:
                img = cv2.imread(os.path.join(test_img_path, img_file))
                if img.shape != (460, 700, 3):
                    img = cv2.resize(img, (700, 460), interpolation=cv2.INTER_CUBIC)
                    print('new image size', img.shape)

                test_imgs_dataset.append(img)

                tokens = img_file.split("_")
                if tokens[1] == 'B':
                    label = 0
                elif tokens[1] == 'M':
                    label = 1
                else:
                    raise ValueError('Unexpected Label!')
                test_label_dataset.append(label)

            # convert to numpy array
            train_imgs_dataset = np.stack(train_imgs_dataset).astype(np.uint8)
            train_label_dataset = np.stack(train_label_dataset).astype(np.uint8)
            test_imgs_dataset = np.stack(test_imgs_dataset).astype(np.uint8)
            test_label_dataset = np.stack(test_label_dataset).astype(np.uint8)

            # save data
            save_path = os.path.join(data_path, 'hdf5', 'fold' + str(i + 1))
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_train_file = os.path.join(save_path, 'train_' + mag + '.h5')
            hf = h5py.File(save_train_file, 'w')
            hf.create_dataset('images', data=train_imgs_dataset)
            hf.create_dataset('labels', data=train_label_dataset)
            hf.close()

            save_test_file = os.path.join(save_path, 'test_' + mag + '.h5')
            hf = h5py.File(save_test_file, 'w')
            hf.create_dataset('images', data=test_imgs_dataset)
            hf.create_dataset('labels', data=test_label_dataset)
            hf.close()

            print('saved fold', str(i + 1), mag)


if __name__ == '__main__':
    main()
