#import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from random import shuffle
from tqdm import tqdm
from PIL import Image

import tensorflow as tf

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import data_helpers

IMG_SIZE = data_helpers.SIZE[0]
LR = 1e-3
MODEL_NAME = 'dogsVScats-{}-{}.model'.format(LR, '6conv')

MODEL_LODED = False


def create_model():
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    conv1 = conv_2d(convnet, 32, 2, activation='relu', name='conv1')
    conv1_ = max_pool_2d(conv1, 2)

    conv2 = conv_2d(conv1_, 64, 2, activation='relu', name='conv2')
    conv2_ = max_pool_2d(conv2, 2)

    conv3 = conv_2d(conv2_, 32, 2, activation='relu', name='conv3')
    conv3_ = max_pool_2d(conv3, 2)

    conv4 = conv_2d(conv3_, 64, 2, activation='relu', name='conv4')
    conv4_ = max_pool_2d(conv4, 2)

    conv5 = conv_2d(conv4_, 32, 2, activation='relu', name='conv5')
    conv5_ = max_pool_2d(conv5, 2)

    conv6 = conv_2d(conv5_, 64, 2, activation='relu', name='conv6')
    conv6_ = max_pool_2d(conv6, 2)

    fc1 = fully_connected(conv6_, 1024, activation='relu', name='fc1')
    fc1_ = dropout(fc1, 0.8)

    fc2 = fully_connected(fc1_, 2, activation='softmax', name='fc2')
    fc2_ = regression(fc2, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(fc2_, tensorboard_dir='log')
    return model


def trainModel(model, X_train, Y_train, X_test, Y_test):
    model.fit({'input': X_train}, {'targets': Y_train}, n_epoch=200, validation_set=({'input': X_test}, {'targets': Y_test}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    model.save(MODEL_NAME)


def testModel(imgdir=data_helpers.CHALLENGEDIR):
    global MODEL_LODED
    global model
    
    if (MODEL_LODED == False):
        model = create_model()
        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('Model Loaded!')
            MODEL_LODED = True
        else:
            return

    X_test1, ID_test1 = data_helpers.load_CNN_test1(numimgs=12)
    result = model.predict(X_test1)

    fig = plt.figure(figsize=(8.5, 6.4))
    for i in range(len(ID_test1)):
        y = fig.add_subplot(3, 4, i+1)
        img = Image.open(os.path.join(imgdir, str(ID_test1[i])+'.jpg')).resize((200, 200))
        y.imshow(img)

        # cat: [1, 0]
        # dog: [0, 1]
        if np.argmax(result[i]) == 1:
            str_label = 'ID: {:d}, Dog'.format(ID_test1[i])
        else:
            str_label = 'ID: {:d}, Cat'.format(ID_test1[i])

        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.tight_layout(pad=1.5)
    plt.show()

def challenge(imgdir=data_helpers.CHALLENGEDIR):
    global MODEL_LODED
    global model
    
    if (MODEL_LODED == False):
        model = create_model()
        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('Model Loaded!')
            MODEL_LODED = True
        else:
            return

    X_test1, ID_test1 = data_helpers.load_CNN_test1()
    sorted_test1 = sorted(zip(ID_test1, X_test1))

    with open('submission-file.csv', 'w') as f:
        f.write('id,label\n')
        for i in tqdm(range(len(sorted_test1))):
            X = sorted_test1[i][1].reshape((1, IMG_SIZE, IMG_SIZE, 3))
            result = model.predict(X)[0]

            f.write('{}, {}\n'.format(sorted_test1[i][0], result[1]))

def main():

    X_train, Y_train, X_test, Y_test = data_helpers.load_data_CNN()
 
    model = create_model()
    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('Model Loaded!')


    trainModel(model, X_train, Y_train, X_test, Y_test)


if __name__ == "__main__":
    main()
