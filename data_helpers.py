#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image

IMGDIR = "./all/train/"
CHALLENGEDIR = "./all/test1/"
DATADIR = "./data/"
SIZE = (32, 32)

def get_image_list(imgdir=IMGDIR):
    image_list = [f for f in os.listdir(imgdir) if os.path.isfile(os.path.join(imgdir, f))]
    if '.DS_Store' in image_list:
        image_list.remove('.DS_Store')
    return image_list

def pre_process(imgdir=IMGDIR, size=SIZE):
    image_list = get_image_list(imgdir)
    if not os.path.isdir(os.path.join(imgdir, 'preprocess')):
        os.mkdir(os.path.join(imgdir, 'preprocess'))
    for image in image_list:
        img = Image.open(os.path.join(imgdir, image))
        img = img.resize(size, Image.LANCZOS)
        img.save(os.path.join(imgdir, 'preprocess', image), 'JPEG')

def gen_test(image_list=get_image_list(), ratio=0.2):
    test_list = list(np.random.choice(image_list, size=int(len(image_list)*ratio), replace=False))
    for test in test_list:
        image_list.remove(test)
    return image_list, test_list

def load(load_list, imgdir=os.path.join(IMGDIR, 'preprocess'), size=SIZE):
    if not os.path.isdir(imgdir):
        print('{} does not exit.'.format(imgdir))
        return

    X = np.zeros((len(load_list), size[0]*size[1]*3))
    Y = np.zeros(len(load_list))
    for i in range(len(load_list)):
        img = Image.open(os.path.join(imgdir, load_list[i]))
        if (img.size != SIZE):
            print('{} is not {:d}x{:d}.'.format(load_list[i], size[0], size[1]))
            return
        X[i] = np.array(img.getdata()).flatten()
        if 'dog' in load_list[i]:
            Y[i] = 1

    return X, Y

def load_data():
    train_list, test_list = gen_test()
    X_train, Y_train = load(train_list)
    X_test, Y_test = load(test_list)
    return X_train, Y_train, X_test, Y_test

def load_CNN(load_list, imgdir=os.path.join(IMGDIR, 'preprocess'), size=SIZE):
    if not os.path.isdir(imgdir):
        print('{} does not exit.'.format(imgdir))
        return

    X = np.zeros((len(load_list), size[0], size[1], 3))
    Y = np.zeros((len(load_list), 2))
    for i in range(len(load_list)):
        img = Image.open(os.path.join(imgdir, load_list[i]))
        if (img.size != SIZE):
            print('{} is not {:d}x{:d}.'.format(load_list[i], size[0], size[1]))
            return

        X[i] = np.array(img.getdata()).reshape((-1, size[0], size[1], 3))

        if 'dog' in load_list[i]:
            Y[i, 1] = 1
        elif 'cat' in load_list[i]:
            Y[i, 0] = 1

    return X, Y

def load_data_CNN(datadir=DATADIR, size=SIZE):
    X_train_npy = os.path.join(datadir, "X_train_"+str(size[0])+".npy")
    Y_train_npy = os.path.join(datadir, "Y_train_"+str(size[0])+".npy")
    X_test_npy = os.path.join(datadir, "X_test_"+str(size[0])+".npy")
    Y_test_npy = os.path.join(datadir, "Y_test_"+str(size[0])+".npy")
    train_list, test_list = gen_test()
    
    if ( os.path.isfile(X_train_npy) and os.path.isfile(Y_train_npy) ):
        X_train = np.load(X_train_npy)
        Y_train = np.load(Y_train_npy)
    else:
        X_train, Y_train = load_CNN(train_list)
        np.save(X_train_npy, X_train)
        np.save(Y_train_npy, Y_train)
    
    if ( os.path.isfile(X_test_npy) and os.path.isfile(Y_test_npy) ):
        X_test = np.load(X_test_npy)
        Y_test = np.load(Y_test_npy)
    else:
        X_test, Y_test = load_CNN(train_list)
        np.save(X_test_npy, X_test)
        np.save(Y_test_npy, Y_test)
    
    return X_train, Y_train, X_test, Y_test

def load_CNN_test1(numimgs=-1, imgdir=os.path.join(CHALLENGEDIR, 'preprocess'), size=SIZE):
    if not os.path.isdir(imgdir):
        print('{} does not exit.'.format(imgdir))
        return
    
    load_list = get_image_list(imgdir)
    if ((numimgs != -1) and (numimgs > 0)):
        load_list = np.random.choice(load_list, numimgs, replace=False)
    elif (numimgs != -1):
        print('Incorrect parameter numimgs.')
        return
    
    X = np.zeros((len(load_list), size[0], size[1], 3))
    ID = np.zeros(len(load_list), dtype=np.int)
    for i in range(len(load_list)):
        img = Image.open(os.path.join(imgdir, load_list[i]))
        if (img.size != SIZE):
            print('{} is not {:d}x{:d}.'.format(load_list[i], size[0], size[1]))
            return
    
        X[i] = np.array(img.getdata()).reshape((-1, size[0], size[1], 3))
        ID[i] = int(load_list[i].split('.')[0])
    
    return X, ID

def gen_batch(data, batch_size, num_iter):
  data = np.array(data)
  index = len(data)
  for i in range(num_iter):
    index += batch_size
    if (index + batch_size > len(data)):
      index = 0
      shuffled_indices = np.random.permutation(np.arange(len(data)))
      data = data[shuffled_indices]
    yield data[index:index + batch_size]
