import os 
from tqdm import tqdm
import nibabel as nib
import numpy as np
import glob
import matplotlib.pyplot as plt
from natsort import natsorted
from tqdm import tqdm

import numpy
from PIL import Image

BASE_PATH = '/mnt/md0/MICCAI_BRATS2020'
Dataset_PATH = os.path.join(BASE_PATH, 'Dataset')
train_path = os.path.join(Dataset_PATH, 'train')

# num = len(os.listdir(train_path))
# print("num:", num) # 369

new_Dataset_PATH = os.path.join(BASE_PATH, 'Dataset_png')
if(not os.path.exists(new_Dataset_PATH)):
    os.mkdir(new_Dataset_PATH)

new_train_path = os.path.join(new_Dataset_PATH, 'train')
if(not os.path.exists(new_train_path)):
    os.mkdir(new_train_path)

new_validation_path = os.path.join(new_Dataset_PATH, 'validation')
if(not os.path.exists(new_validation_path)):
    os.mkdir(new_validation_path)

new_test_path = os.path.join(new_Dataset_PATH, 'test')
if(not os.path.exists(new_test_path)):
    os.mkdir(new_test_path)

id_num = 0

for patient_id in tqdm(natsorted(os.listdir(train_path))):
    patient_path = os.path.join(train_path, patient_id)
    t1_image_dir = os.path.join(patient_path, 't1.npy')
    t1_image = np.load(t1_image_dir)

    mask_dir = os.path.join(patient_path, 'mask.npy')
    mask = np.load(mask_dir)

    # print("t1_image.shape[1]:", t1_image.shape[1])

    for slice_num in range(t1_image.shape[1]):

        mask_sum = np.sum(mask[:,slice_num,:])

        if(mask_sum > 0.):
            label = 1
        else:
            label = 0

        t1_image_slice = t1_image[:,slice_num,:]
        image_array = numpy.asarray(t1_image_slice)

        image_array_copy = numpy.copy(image_array)

        flattened_image = image_array_copy.flatten()

        if(max(flattened_image) == 0):
            # new_image_array = image_array_copy
            continue
        else:
            new_image_array = image_array_copy / max(flattened_image)

        new_image_array_rotate = Image.fromarray(numpy.uint8(new_image_array * 255))
        new_image_array_rotate = new_image_array_rotate.crop((0, 30, 155, 210))
        new_image_array_rotate = new_image_array_rotate.rotate(90)
        new_image_array_rotate = new_image_array_rotate.resize((256,256))
        new_image_array_rotate = np.array(new_image_array_rotate) / 255.

        new_image_array_rotate = np.expand_dims(new_image_array_rotate, axis=-1)
        saved_image = np.concatenate((new_image_array_rotate,new_image_array_rotate,new_image_array_rotate), axis=-1)

        im = Image.fromarray(numpy.uint8(saved_image * 255))

        if(id_num < 300):
            new_patient_path = os.path.join(new_train_path, patient_id)
        elif(id_num < 335):
            new_patient_path = os.path.join(new_validation_path, patient_id)
        else:
            new_patient_path = os.path.join(new_test_path, patient_id)

        if(not os.path.exists(new_patient_path)):
            os.mkdir(new_patient_path)

        im.save(os.path.join(new_patient_path, "slice_"+str(slice_num)+"_"+str(label)+".png"))

    id_num += 1