from tensorflow.keras.datasets import mnist,cifar10,cifar100
import numpy as np

def get_mnist():
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    x_train = x_train/255.0
    x_test = x_test/255.0
    x_train = np.reshape(x_train,(-1,28,28,1))
    x_test = np.reshape(x_test,(-1,28,28,1))

    return x_train,y_train,x_test,y_test

def get_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test

def get_imagenet():
    x_train = np.load('G:/datasets_and_networks/imagenet/data/x_train.npy')
    x_test = np.load('G:/datasets_and_networks/imagenet/data/x_test.npy')
    y_train = np.load('G:/datasets_and_networks/imagenet/data/y_train.npy')
    y_test = np.load('G:/datasets_and_networks/imagenet/data/y_test.npy')

    return x_train, y_train, x_test, y_test