import os
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras import layers

import sklearn
from sklearn.model_selection import train_test_split

from natsort import natsorted

import cv2
import shutil
import glob


image_shape = (224,224,3)

leaky_relu_ratio = 0.01
dropout_ratio = 0.2

base_model = tf.keras.applications.InceptionResNetV2(include_top=False, input_shape=image_shape, weights='imagenet')

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(LeakyReLU(leaky_relu_ratio))
model.add(Dropout(dropout_ratio))
model.add(Dense(128))
model.add(BatchNormalization())
model.add(LeakyReLU(leaky_relu_ratio))
model.add(Dropout(dropout_ratio))
model.add(Dense(1, activation = 'sigmoid'))

learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate)
# decayed_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, 1000, 0.96)
# optimizer = tf.keras.optimizers.Adam(decayed_learning_rate)

checkpoint_dir = './binary_classification'
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 model=model)

ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=10)

checkpoint.restore(ckpt_manager.latest_checkpoint)

dataset_dir = '/mnt/md0/MICCAI_BRATS2020/Dataset_png'

def get_label(file_name):
    label_tag = file_name.split('.')[0].split('_')[-1]
    if(label_tag == '0'):
        return 'No'
    else:
        return 'Yes'

def dataset_collection_func():

    train_filepaths=[]
    train_labels=[]
    validation_filepaths=[]
    validation_labels=[]
    test_filepaths=[]
    test_labels=[]

    train_dir = os.path.join(dataset_dir, 'train')
    validation_dir = os.path.join(dataset_dir, 'validation')
    test_dir = os.path.join(dataset_dir, 'test')

    # train
    for patient_item in tqdm(natsorted(os.listdir(train_dir))):
        patient_path_saved = os.path.join(train_dir, patient_item)

        slice_path_list = os.listdir(patient_path_saved)

        for slice_num in range(len(slice_path_list)):
            train_filepaths.append(os.path.join(patient_path_saved, slice_path_list[slice_num]))
            train_labels.append(get_label(slice_path_list[slice_num]))
    
    # validation
    for patient_item in tqdm(natsorted(os.listdir(validation_dir))):
        patient_path_saved = os.path.join(validation_dir, patient_item)

        slice_path_list = os.listdir(patient_path_saved)

        for slice_num in range(len(slice_path_list)):
            validation_filepaths.append(os.path.join(patient_path_saved, slice_path_list[slice_num]))
            validation_labels.append(get_label(slice_path_list[slice_num]))

    # test
    for patient_item in tqdm(natsorted(os.listdir(test_dir))):
        patient_path_saved = os.path.join(test_dir, patient_item)

        slice_path_list = os.listdir(patient_path_saved)

        for slice_num in range(len(slice_path_list)):
            test_filepaths.append(os.path.join(patient_path_saved, slice_path_list[slice_num]))
            test_labels.append(get_label(slice_path_list[slice_num]))

    train_filepaths = np.array(train_filepaths)
    train_labels = np.array(train_labels)
    validation_filepaths = np.array(validation_filepaths)
    validation_labels = np.array(validation_labels)
    test_filepaths = np.array(test_filepaths)
    test_labels = np.array(test_labels)

    # print(train_labels[:10])
    train_labels = np.where(train_labels=='No', 0, 1)
    # print(train_labels[:10])
    validation_labels = np.where(validation_labels=='No', 0, 1)
    test_labels = np.where(test_labels=='No', 0, 1)

    return train_filepaths, train_labels, validation_filepaths, validation_labels, test_filepaths, test_labels

def get_image(img_path):
    img_array = Image.open(img_path)
    img_array.load()
    img_array = img_array.resize((image_shape[0],image_shape[1]))
    img_array = np.asarray(img_array).astype(np.float32)
    # print("img_array.shape:", img_array.shape)

    img_array = img_array / 255.
    img_array = img_array - np.mean(img_array)

    return img_array

def CE_loss(true_label, prediction):
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    loss = cross_entropy(true_label, prediction)
    return loss

def my_metrics(y_true, y_pred):
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    y_pred = np.where(y_pred >= 0.5, 1, 0)

    TP, TN, FP, FN = 0, 0, 0, 0
    for prediction, y in zip(y_pred, y_true):

        if(prediction == y):
            if(prediction == 1): # {'No': 0, 'Yes': 1}
                TP += 1
            else:
                TN += 1
        else:
            if(prediction == 1):
                FP += 1
            else:
                FN += 1

    precision = TP/(TP+FP+1.0e-4)

    recall = TP/(TP+FN+1.0e-4)

    f_measure = (2. * precision * recall)/(precision + recall + 1.0e-4)

    accuracy = (TP + TN) / (TP + TN + FP + FN+1.0e-4)

    # print("TP:", TP)
    # print("TN:", TN)
    # print("FP:", FP)
    # print("FN:", FN)

    # print("precision:", precision)
    # print("recall:", recall)
    # print("f_measure:", f_measure)
    # print("accuracy:", accuracy)

    return np.array([TP, TN, FP, FN, precision, recall, f_measure, accuracy])

def train_step(inputs, labels, optimizer):
    # print("training......")

    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = CE_loss(labels, predictions)
        # print(loss)
    
    # labels = tf.make_tensor_proto(labels)
    # predictions = tf.make_tensor_proto(predictions)
    metric_results = my_metrics(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return np.array(loss), metric_results, optimizer

def valid_step(inputs, labels):
    # print("validation......")

    predictions = model(inputs, training=False)
    loss = CE_loss(labels, predictions)
    # print(loss)

    metric_results = my_metrics(labels, predictions)

    return np.array(loss), metric_results, predictions

# add tensorboard
def train(train_filepaths, train_labels, validation_filepaths, validation_labels, epochs, BATCH_SIZE):

    global learning_rate
    global optimizer

    best_auc_roc = 0

    best_val_loss = 0
    best_val_loss_epoch_lr_decay = 0
    best_val_loss_epoch = 0

    first_tag = 1

    iteration = 0
    val_iteration = 0

    for epoch in range(epochs):
        start = time.time()

        idx = np.random.permutation(len(train_labels))
        train_filepaths, train_labels = train_filepaths[idx], train_labels[idx]

        print("train epoch = ", epoch)
        for index in range(0, len(train_labels)-BATCH_SIZE, BATCH_SIZE):
            label_batch = [] # always load the same batch
            for i in range(BATCH_SIZE):
                img_path = train_filepaths[index+i]

                img_array = get_image(img_path)

                # data augmentation
                img_array = tf.keras.preprocessing.image.random_rotation(img_array, 0.2)
                img_array = tf.keras.preprocessing.image.random_shift(img_array, 0.1, 0.1)
                img_array = tf.keras.preprocessing.image.random_shear(img_array, 0.1)
                img_array = tf.keras.preprocessing.image.random_zoom(img_array, (0.7,1))

                img_array = np.array(img_array)
                img_array = np.expand_dims(img_array, axis=0)

                if(i == 0):
                    image_batch = img_array
                else:
                    image_batch = np.concatenate((image_batch, img_array), axis=0)

                label_batch.append(train_labels[index+i])

            label_batch = np.array(label_batch)
            label_batch = np.expand_dims(label_batch, axis=-1)

            loss, metric_results, optimizer = train_step(image_batch, label_batch, optimizer)

            # if(iteration % 30 == 0):
            if(iteration % 300 == 0):

                idx_val = np.random.permutation(len(validation_labels))
                validation_filepaths, validation_labels = validation_filepaths[idx_val], validation_labels[idx_val]

                val_loss_average = 0
                val_tag = 0

                accuracy_average = 0
                recall_average = 0
                precision_average = 0

                all_val_labels = []
                all_val_predictions = []

                print("validation")
                # validation
                for val_index in range(0, len(validation_labels)-BATCH_SIZE, BATCH_SIZE):
                    label_batch = [] # always load the same batch
                    for i in range(BATCH_SIZE):
                        img_path = validation_filepaths[val_index+i]

                        img_array = get_image(img_path)

                        img_array = np.expand_dims(img_array, axis=0)

                        if(i == 0):
                            image_batch = img_array
                        else:
                            image_batch = np.concatenate((image_batch, img_array), axis=0)

                        label_batch.append(validation_labels[val_index+i])

                    label_batch = np.array(label_batch)
                    label_batch = np.expand_dims(label_batch, axis=-1)

                    loss, metric_results, val_predictions = valid_step(image_batch, label_batch)

                    val_tag += 1
                    val_loss_average += loss

                    accuracy_average += metric_results[7]
                    recall_average += metric_results[5]
                    precision_average += metric_results[4]

                    all_val_labels.append(label_batch.flatten())
                    all_val_predictions.append(np.array(val_predictions).flatten())
                    # AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'flatten'

                val_loss_average = val_loss_average / val_tag

                accuracy_average = accuracy_average / val_tag
                recall_average = recall_average / val_tag
                precision_average = precision_average / val_tag

                all_val_labels = np.array(all_val_labels)
                all_val_predictions = np.array(all_val_predictions)

                all_val_labels = all_val_labels.flatten()
                all_val_predictions = all_val_predictions.flatten()

                auc_roc_average = sklearn.metrics.roc_auc_score(all_val_labels, all_val_predictions)
                # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.

                print("val_loss_average:", val_loss_average)
                print("accuracy_average:", accuracy_average)
                print("recall_average:", recall_average)
                print("precision_average:", precision_average)
                print("auc_roc_average:", auc_roc_average)
                
                # save the best only
                # best val_auc_roc
                if(first_tag == 1 or auc_roc_average > best_auc_roc):
                    best_auc_roc = auc_roc_average
                    print("saveing model")
                    ckpt_manager.save()

                    f = open("best_auc_roc_binary_classification.txt", "w")
                    f.write(str(best_auc_roc))
                    f.close()
                
                # best val_loss
                if(first_tag == 1 or val_loss_average < best_val_loss):
                    best_val_loss = val_loss_average
                    best_val_loss_epoch = val_iteration
                    best_val_loss_epoch_lr_decay = best_val_loss_epoch
                    # print("saveing model")
                    # ckpt_manager.save()

                # learning rate decay --> reduce learning rate on plateau
                plateau_patience = 30

                if(val_loss_average > best_val_loss and (val_iteration - best_val_loss_epoch_lr_decay) == plateau_patience):

                    checkpoint.restore(ckpt_manager.latest_checkpoint)

                    best_val_loss_epoch_lr_decay = val_iteration
                    print("learning rate on plateau, reduce learning rate")
                    decay_factor = 0.5
                    learning_rate = learning_rate * decay_factor
                    optimizer = tf.keras.optimizers.Adam(learning_rate)

                # early stopping
                early_stopping_patience = 150
                if(val_loss_average > best_val_loss and (val_iteration - best_val_loss_epoch) == early_stopping_patience):
                    print("early stopping")
                    return

                first_tag = 0

                val_iteration += 1

            iteration += 1

def evaluation(test_filepaths, test_labels):
    print("testing")

    checkpoint.restore(ckpt_manager.latest_checkpoint)

    labels = []
    predictions = []

    for i in range(len(test_labels)):
        img_path = test_filepaths[i]

        img_array = get_image(img_path)

        img_array = np.expand_dims(img_array, axis=0)

        image_batch = img_array

        prediction = model(image_batch, training=False)
        prediction = tf.make_tensor_proto(prediction).float_val
        predictions.append(prediction[0])
        
        labels.append(test_labels[i])

    labels = np.array(labels)
    predictions = np.array(predictions)

    metric_results = my_metrics(labels, predictions)

    auc_roc = sklearn.metrics.roc_auc_score(labels, predictions)

    print("TP:", metric_results[0])
    print("TN:", metric_results[1])
    print("FP:", metric_results[2])
    print("FN:", metric_results[3])

    print("precision:", metric_results[4])
    print("recall:", metric_results[5])
    print("f_measure:", metric_results[6])
    print("accuracy:", metric_results[7])
    print("auc_roc:", auc_roc)

epochs = 200
BATCH_SIZE = 16
train_filepaths, train_labels, validation_filepaths, validation_labels, test_filepaths, test_labels = dataset_collection_func()
# train(train_filepaths, train_labels, validation_filepaths, validation_labels, epochs, BATCH_SIZE)
evaluation(validation_filepaths, validation_labels)
evaluation(test_filepaths, test_labels)
