import torch
import torch.nn as nn
import torch.nn.parallel
import argparse
import torch.utils.data
import os
import numpy as np
from math import trunc
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from dataset import BreaKHisOnline, BreaKHisOnlineTestPatient
import math
import time
import logging
# from tensorboardX import SummaryWriter
from utils.get_mean_std import get_mean_std_across_mag
from model import vgg19_bn
from utils.rename_dict import get_new_names, rename_dict
from collections import OrderedDict
from utils.get_patient_ids import get_patient2id, get_id2patient, get_patient_gt


parser = argparse.ArgumentParser()
parser.add_argument('--gpus', default='0', help='number of GPUs used', type=str)
parser.add_argument('--fold', default='3', help='which fold', type=int)


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

# global settings
fold = args.fold
mag = '40X'  # 40X, 100X, 200X, 400X
mean, std = get_mean_std_across_mag(fold)

batch_size = 8
num_workers = 1
num_epoch = 130
resume_epoch = 0
resume = False
weight_decay = 1e-4

lr = 0.00005  # 0.0001 for adam, 0.001 for SGD
gamma = 0.5  # 0.5 for adam, 0.1 for SGD
milestone = [30, 60]
resize_img_size = [224, 224]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set data loader
# angles = np.array([36, 45, 72, 90, 108, 135, 144, 180, 216, 225, 252, 270, 288, 315, 324]) / 180 * math.pi
angles = [-math.pi, math.pi]
train_dataset = BreaKHisOnline(fold=fold, split='train', transform=transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(angles),
    transforms.Resize(resize_img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)  # it seems like the mean and std here have no effect on the final performance.
                                     # Can be commented out
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# ### test data ###
test_dataset_40X = BreaKHisOnlineTestPatient(fold=fold, mag='40X', transform=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(resize_img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))
test_dataloader_40X = torch.utils.data.DataLoader(test_dataset_40X, batch_size=batch_size, num_workers=num_workers)
test_40X_patients = test_dataset_40X.get_patient_set()

test_dataset_100X = BreaKHisOnlineTestPatient(fold=fold, mag='100X', transform=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(resize_img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))
test_dataloader_100X = torch.utils.data.DataLoader(test_dataset_100X, batch_size=batch_size, num_workers=num_workers)
test_100X_patients = test_dataset_100X.get_patient_set()

test_dataset_200X = BreaKHisOnlineTestPatient(fold=fold, mag='200X', transform=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(resize_img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))
test_dataloader_200X = torch.utils.data.DataLoader(test_dataset_200X, batch_size=batch_size, num_workers=num_workers)
test_200X_patients = test_dataset_200X.get_patient_set()

test_dataset_400X = BreaKHisOnlineTestPatient(fold=fold, mag='400X', transform=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(resize_img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))
test_dataloader_400X = torch.utils.data.DataLoader(test_dataset_400X, batch_size=batch_size, num_workers=num_workers)
test_400X_patients = test_dataset_400X.get_patient_set()

# get patient id mapping
_mags = ['40X', '100X', '200X', '400X']
patient2id = {}
id2patient = {}
patient_gt = {}

for _mag in _mags:
    patient2id[_mag] = get_patient2id('utils/' + _mag + '_patient.txt')
    id2patient[_mag] = get_id2patient('utils/' + _mag + '_patient.txt')
    patient_gt[_mag] = get_patient_gt('utils/' + _mag + '_patient.txt')

print('Training set size:', len(train_dataset))
print('40X Test set size:', len(test_dataset_40X))
print('100X Test set size:', len(test_dataset_100X))
print('200X Test set size:', len(test_dataset_200X))
print('400X Test set size:', len(test_dataset_400X))

test_dataloaders = [test_dataloader_40X, test_dataloader_100X, test_dataloader_200X, test_dataloader_400X]

# specify model and log output directory
root_out_dir = '/media/pwu/Data/saved_models/pathology/across_mag'
model_out_dir = os.path.join(root_out_dir)
try:
    os.makedirs(model_out_dir)
except OSError:
    pass

########################################
# specify logger
log_out_dir = os.path.join(root_out_dir, 'logger')
try:
    os.makedirs(log_out_dir)
except OSError:
    pass

log_dir = os.path.join(log_out_dir, 'log-' + time.strftime("%Y%m%d-%H%M%S") + '.txt')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    filename=log_dir,
                    filemode='w')

# tensorboard logger
# writer = SummaryWriter()

########################################
# build model
classifier = vgg19_bn()

if resume is True:
    model_path = os.path.join(root_out_dir, 'cls_model_' + str(resume_epoch) + '.pth')
    classifier.load_state_dict(torch.load(model_path))
else:
    # load existing pre-trained models
    pre_trained_model_path = '/media/pwu/Data/saved_models/pathology/vgg19_bn.pth'
    pre_trained_dict = torch.load(pre_trained_model_path)

    model_dict = classifier.state_dict()

    # get the new key names of pre_trained_dict
    keys = list(pre_trained_dict.keys())
    feature_keys = list(filter(lambda x: x.startswith('features'), keys))
    # classifier_keys = list(filter(lambda x: x.startswith(('classifier.0', 'classifier.3')), keys))

    # get new name: [float('-inf'), 5, 14, 27, 40, float('inf')]
    # the second parameter of get_new_names should be adjusted based on the specific network configuration
    new_keys = get_new_names(feature_keys, [float('-inf'), 5, 14, float('inf')], [0, 1, 2])
    # new_keys += classifier_keys

    # rename and filter the pre_trained_dict
    # required_weights = ('features', 'classifier.0', 'classifier.3')
    required_weights = 'features'
    pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if str(k).startswith(required_weights)}
    pre_trained_dict = OrderedDict(pre_trained_dict)
    pre_trained_dict = rename_dict(pre_trained_dict, new_keys)

    model_dict.update(pre_trained_dict)

    classifier.load_state_dict(model_dict)

########################################
# define optimizer
# set different learning rates for different layers
# params = []
# for key, value in dict(classifier.named_parameters()).items():
#     if key[:10] == 'classifier':
#         params += [{'params': value, 'lr': 0.5 * lr}]
#     else:
#         params += [{'params': value}]

optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=False)
# optimizer = optim.Adam(params, lr=lr, weight_decay=1e-4)
classifier.to(device)

# scheduler
exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestone, gamma=gamma)

num_batch = len(train_dataset)/batch_size

if resume:
    start_epoch = resume_epoch + 1
else:
    start_epoch = 0

mags = ['40X', '100X', '200X', '400X']

########################################
# start training
for epoch in range(start_epoch, num_epoch):
    train_correct = 0
    train_loss = 0

    num_train_data = 0

    exp_lr_scheduler.step()
    classifier.train()

    for i, data in enumerate(train_dataloader):
        images, targets = data
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        pred = classifier(images)
        loss = F.nll_loss(pred, targets)

        loss.backward()
        optimizer.step()

        pred_choice = pred.data.max(1)[1]
        train_correct = train_correct + pred_choice.eq(targets.data).cpu().sum().numpy()
        train_loss = train_loss + loss.data.item()
        num_train_data = num_train_data + targets.data.cpu().numpy().size

        msg = '[{0:d}: {1:d}/{2:d}] accuracy: {3:f}'.format(
            epoch, i, trunc(num_batch), train_correct / float(num_train_data))
        logging.info(msg)
        print(msg)

    # === Tensorboard logging === #
    # writer.add_scalar('data/loss', train_loss, epoch)
    # writer.add_scalar('data/train_accuracy', train_correct / float(num_train_data), epoch)

    # for name, param in classifier.named_parameters():
    #     writer.add_histogram(name, param.clone().to('cpu').data.numpy(), epoch)

    if epoch % 5 == 0:
        torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (root_out_dir, epoch))


# evaluate
classifier.eval()

test_correct = np.zeros(4, np.int32)
test_correct_patients = np.zeros(4, np.int32)

num_test_data = np.zeros(4, np.int32)
num_test_data_patients = np.array([test_dataset_40X.get_patient_num(), test_dataset_100X.get_patient_num(),
                                   test_dataset_200X.get_patient_num(), test_dataset_400X.get_patient_num()])

# dict for patient-level statistics
patient_40X_vote = {el: 0 for el in test_40X_patients}
patient_100X_vote = {el: 0 for el in test_100X_patients}
patient_200X_vote = {el: 0 for el in test_200X_patients}
patient_400X_vote = {el: 0 for el in test_400X_patients}
patient_votes = [patient_40X_vote, patient_100X_vote, patient_200X_vote, patient_400X_vote]

for t, test_dataloader in enumerate(test_dataloaders):
    for i, data in enumerate(test_dataloader):
        images, targets, patient_id = data
        images, targets = images.to(device), targets.to(device)

        pred = classifier(images)

        pred_choice = pred.data.max(1)[1]
        test_correct[t] = test_correct[t] + pred_choice.eq(targets.data).cpu().sum().numpy()
        num_test_data[t] = num_test_data[t] + targets.data.cpu().numpy().size

        # majority voting for patient-level classification
        patient_id = patient_id.numpy()
        pred_choice = pred_choice.cpu().numpy()
        for pid, pred in zip(patient_id, pred_choice):
            patient_name = id2patient[mags[t]][pid]
            if pred == 0:  # benign
                patient_votes[t][patient_name] += 1
            else:
                patient_votes[t][patient_name] -= 1

    curr_accuracy = test_correct[t] / float(num_test_data[t])
    print("{} test correct {}".format(mags[t], test_correct[t]))
    print("{} num test data {}".format(mags[t], num_test_data[t]))

    # compute patient-level accuracy
    votes = patient_votes[t]
    for k, v in votes.items():
        if v >= 0:
            pred_label = 0
        else:
            pred_label = 1

        if pred_label == patient_gt[mags[t]][k]:
            test_correct_patients[t] += 1

    curr_accuracy_patients = test_correct_patients[t] / float(num_test_data_patients[t])
    print("{} test correct patient {}".format(mags[t], test_correct_patients[t]))
    print("{} num test data patient {}".format(mags[t], num_test_data_patients[t]))

    # === Tensorboard logging === #
    # writer.add_scalar('data/' + mags[t] + ' test_accuracy', curr_accuracy, epoch)

    msg = '*** {} Test accuracy: {}'.format(mags[t], curr_accuracy)
    msg2 = '*** {} Test accuracy patient: {}'.format(mags[t], curr_accuracy_patients)

    logging.info(msg)
    logging.info(msg2)

    print(msg)
    print(msg2)

# writer.close()
