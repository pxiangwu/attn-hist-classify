from __future__ import print_function
import torch
from torch.utils.data.dataset import Dataset
import h5py
import numpy as np
import os
import cv2
from utils.get_patient_ids import get_patient2id


class BreaKHis(Dataset):
    def __init__(self, fold=1, mag='40X', split='train', root_data_path=None, transform=None):
        if root_data_path is None:
            root_data_path = '/media/pwu/Data/2D_data/pathology_images/mkfold/hdf5'

        self.transform = transform

        data_path = os.path.join(root_data_path, 'fold' + str(fold))
        data_file = os.path.join(data_path, split + '_' + mag + '.h5')

        hf = h5py.File(data_file, 'r')
        self.data = np.array(hf.get('images'))
        self.labels = np.array(hf.get('labels'))
        hf.close()

    def __getitem__(self, index):
        label = np.array(self.labels[index]).astype(np.int64)
        label = torch.from_numpy(label)

        image = self.data[index, ...]
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.data.shape[0]


class BreaKHisOnline(Dataset):
    def __init__(self, fold=1, split='train', root_data_path=None, transform=None):
        if root_data_path is None:
            root_data_path = '/media/pwu/Data/2D_data/pathology_images/mkfold/'

        self.transform = transform

        self.data_path = os.path.join(root_data_path, 'fold' + str(fold), split)

        mags = ['40X', '100X', '200X', '400X']

        self.image_list = []

        for mag in mags:
            current_dir = os.path.join(self.data_path, mag)
            imgs = [os.path.join(mag, f) for f in os.listdir(current_dir) if os.path.isfile(os.path.join(current_dir, f))]

            self.image_list += imgs

        self.data_size = len(self.image_list)

    def __getitem__(self, index):
        image_file_name = self.image_list[index]
        image_path = os.path.join(self.data_path, image_file_name)

        image = cv2.imread(image_path)
        if image.shape != (460, 700, 3):
            image = cv2.resize(image, (700, 460), interpolation=cv2.INTER_CUBIC)

        if self.transform is not None:
            image = self.transform(image)

        # get the label for this image
        tokens = image_file_name.split("_")
        if tokens[1] == 'B':
            label = 0
        else:
            label = 1
        label = np.array(label).astype(np.int64)

        return image, torch.from_numpy(label)

    def __len__(self):
        return self.data_size


class BreaKHisOnlineTestPatient(Dataset):
    def __init__(self, fold=1, mag='40X', root_data_path=None, transform=None):
        if root_data_path is None:
            root_data_path = '/media/pwu/Data/2D_data/pathology_images/mkfold/'

        self.split = 'test'

        self.transform = transform

        # get patient2id
        _mags = ['40X', '100X', '200X', '400X']
        patient2id = {}
        for _mag in _mags:
            patient2id[_mag] = get_patient2id('utils/' + _mag + '_patient.txt')

        self.patient2id_dict = patient2id[mag]

        self.data_path = os.path.join(root_data_path, 'fold' + str(fold), self.split)

        self.image_list = []

        current_dir = os.path.join(self.data_path, mag)
        imgs = [os.path.join(mag, f) for f in os.listdir(current_dir) if os.path.isfile(os.path.join(current_dir, f))]

        self.image_list += imgs

        self.data_size = len(self.image_list)

        # get patient names
        self.patient_names = []

        for img in self.image_list:
            tokens = img.split("_")
            p_tokens = tokens[2].split("-")
            p_name = '-'.join(p_tokens[0:3])

            self.patient_names.append(p_name)

        self.patient_names = list(set(self.patient_names))

    def __getitem__(self, index):
        image_file_name = self.image_list[index]
        image_path = os.path.join(self.data_path, image_file_name)

        image = cv2.imread(image_path)
        if image.shape != (460, 700, 3):
            image = cv2.resize(image, (700, 460), interpolation=cv2.INTER_CUBIC)

        if self.transform is not None:
            image = self.transform(image)

        # get the label for this image
        tokens = image_file_name.split("_")
        if tokens[1] == 'B':
            label = 0
        else:
            label = 1
        label = np.array(label).astype(np.int64)

        # get the patient name for this image
        p_tokens = tokens[2].split("-")
        patient_name = '-'.join(p_tokens[0:3])

        return image, torch.from_numpy(label), self.patient2id_dict[patient_name]

    def __len__(self):
        return self.data_size

    def get_patient_set(self):
        return self.patient_names

    def get_patient_num(self):
        return len(self.patient_names)

