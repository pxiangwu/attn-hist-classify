import os


def patient_ids():
    root_data_path = '/media/pwu/Data/2D_data/pathology_images/mkfold/'
    fold = 1

    imgs = []

    mags = ['40X', '100X', '200X', '400X']

    for mag in mags:
        # train images
        data_path = os.path.join(root_data_path, 'fold' + str(fold), 'train')
        current_dir = os.path.join(data_path, mag)
        imgs += [os.path.join(f) for f in os.listdir(current_dir) if os.path.isfile(os.path.join(current_dir, f))]

        # test images
        data_path = os.path.join(root_data_path, 'fold' + str(fold), 'test')
        current_dir = os.path.join(data_path, mag)
        imgs += [os.path.join(f) for f in os.listdir(current_dir) if os.path.isfile(os.path.join(current_dir, f))]

        patient_gt = get_patient_labels(imgs)

        # save to txt
        f = open(mag + '_patient.txt', 'w+')

        for i, (k, v) in enumerate(patient_gt.items()):
            f.write('{} {} {}\n'.format(k, v, i))

        f.close()


def get_patient_labels(images):
    patient_dict = dict()

    for img in images:
        tokens = img.split('_')
        p_tokens = tokens[2].split('-')

        patient_id = '-'.join(p_tokens[0:3])

        if tokens[1] == 'B':
            patient_label = 0
        else:
            patient_label = 1

        patient_dict[patient_id] = patient_label

    return patient_dict


def get_patient2id(file_path):
    patient2id = dict()

    with open(file_path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            if line.rstrip():
                line = line.rstrip()  # remove '\n'
                line = line.split(' ')

                patient2id[line[0]] = int(line[2])

    return patient2id


def get_id2patient(file_path):
    id2patient = dict()

    with open(file_path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            if line.rstrip():
                line = line.rstrip()  # remove '\n'
                line = line.split(' ')

                id2patient[int(line[2])] = line[0]

    return id2patient


def get_patient_gt(file_path):
    patient_gt = dict()

    with open(file_path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            if line.rstrip():
                line = line.rstrip()  # remove '\n'
                line = line.split(' ')

                patient_gt[line[0]] = int(line[1])

    return patient_gt


if __name__ == '__main__':
    # patient_ids()

    dic = get_patient_gt('40X')
    print(dic)
