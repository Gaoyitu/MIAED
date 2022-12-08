import tensorflow as tf
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,BatchNormalization,ReLU,Reshape,Conv2DTranspose,UpSampling2D,Add,AveragePooling2D,MaxPool2D,Dropout
from tensorflow.keras.models import Sequential
import numpy as np

def get_mnist_local():

    inputs = Input(shape=(28, 28, 1))
    conv_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                                    padding='same', activation="relu", name="conv_1")(inputs)
    max_pooling_1 = MaxPooling2D((2, 2), (2, 2), padding="same")(conv_1)
    conv_2 = Conv2D(64, (3, 3), padding='same', activation="relu", name="conv_2")(max_pooling_1)
    max_pooling_2 = MaxPooling2D((2, 2), (2, 2), padding="same")(conv_2)

    max_pooling_2_flat = Flatten(name='flatten')(max_pooling_2)

    fc_1 = Dense(200, activation="relu",name='feature_layer')(max_pooling_2_flat)

    # outputs = Dense(10, activation='softmax')(fc_1)
    outputs = Dense(10, activation=None)(fc_1)

    model = Model(inputs=inputs, outputs=outputs)

    model.load_weights("networks/mnist/classifiers/mnist_local_weights.h5")
    # model.summary()

    return model

def get_mnist_black():

    inputs = Input(shape=(28, 28, 1))
    conv_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                                    padding='same', activation="relu", name="conv_1")(inputs)
    conv_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                                    padding='same', activation="relu", name="conv_2")(conv_1)
    max_pooling_1 = MaxPooling2D((2, 2), (2, 2), padding="same")(conv_2)
    conv_3 = Conv2D(64, (3, 3), padding='same', activation="relu", name="conv_3")(max_pooling_1)
    conv_4 = Conv2D(64, (3, 3), padding='same', activation="relu", name="conv_4")(conv_3)
    max_pooling_2 = MaxPooling2D((2, 2), (2, 2), padding="same")(conv_4)

    max_pooling_2_flat = Flatten(name='flatten')(max_pooling_2)

    fc_1 = Dense(200, activation="relu",name='fc_1')(max_pooling_2_flat)
    fc_2 = Dense(200, activation="relu", name='fc_2')(fc_1)

    # outputs = Dense(10, activation='softmax')(fc_2)
    outputs = Dense(10, activation=None)(fc_2)

    model = Model(inputs=inputs, outputs=outputs)

    model.load_weights("networks/mnist/classifiers/mnist_black_weights.h5")
    # model.summary()

    return model


def get_cifar10_vgg16():
    conv_base = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    model = Sequential()

    model.add(conv_base)
    model.add(Flatten(name='flatten'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    # model.add(Dense(10, activation="softmax"))
    model.add(Dense(10, activation=None))
    # conv_base.trainable = True

    model.load_weights('networks/cifar10/classifiers/cifar10_vgg16_weights.h5')

    return model,conv_base

def get_cifar10_mobilenet():
    conv_base = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    model = Sequential()

    model.add(conv_base)
    model.add(Flatten(name='flatten'))

    # model.add(tf.keras.layers.Dense(10, activation="softmax"))
    model.add(Dense(10, activation=None))
    # conv_base.trainable = True

    model.load_weights('networks/cifar10/classifiers/cifar10_mobilenet_weights.h5')

    return model,conv_base

def get_cifar10_resnet50():
    conv_base = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    model = Sequential()

    model.add(conv_base)
    model.add(Flatten(name='flatten'))

    # model.add(tf.keras.layers.Dense(10, activation="softmax"))
    # conv_base.trainable = True
    model.add(Dense(10, activation=None))

    model.load_weights('networks/cifar10/classifiers/cifar10_resnet50_weights.h5')

    return model,conv_base

def get_cifar10_local():
    inputs = Input(shape=(32, 32, 3))
    conv_1 = Conv2D(filters=64,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    activation="relu",
                    name="conv_1",
                    kernel_initializer='glorot_uniform')(inputs)
    conv_2 = Conv2D(64, (3, 3),
                    padding='same',
                    activation="relu",
                    name="conv_2",
                    kernel_initializer='glorot_uniform')(conv_1)
    max_pooling_1 = MaxPooling2D((2, 2), (2, 2),
                                  padding="same",name="pool1")(conv_2)
    conv_3 = Conv2D(128, (3, 3),
                    padding='same',
                    activation="relu",
                    name="conv_3",
                    kernel_initializer='glorot_uniform')(max_pooling_1)
    conv_4 = Conv2D(128, (3, 3),
                    padding='same',
                    activation="relu",
                    name="conv_4",
                    kernel_initializer='glorot_uniform')(conv_3)
    max_pooling_2 = MaxPooling2D((2, 2), (2, 2),
                                  padding="same",name="pool2")(conv_4)

    max_pooling_2_flat = Flatten(name='flatten')(max_pooling_2)

    fc_1 = Dense(256,
                 activation="relu",
                 kernel_initializer='he_normal')(max_pooling_2_flat)

    fc_2 = Dense(256,
                 activation="relu",
                 kernel_initializer='he_normal',name = 'fc_2')(fc_1)

    # outputs = Dense(10,activation='softmax')(fc_2)
    outputs = Dense(10,activation=None)(fc_2)

    model = Model(inputs=inputs, outputs=outputs)

    model.load_weights('networks/cifar10/classifiers/cifar10_local_weights.h5')
    return model




def get_imagenet_mobilenet():
    conv_base = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    model = Sequential()

    model.add(conv_base)
    model.add(Flatten(name='flatten'))

    # model.add(Dense(10, activation="softmax"))
    model.add(Dense(10, activation=None))
    conv_base.trainable = False

    model.load_weights('networks/imagenet/classifiers/imagenet_mobilenet_weights.h5')

    return model,conv_base

def get_imagenet_densenet121():
    conv_base = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    model = Sequential()

    model.add(conv_base)
    model.add(Flatten(name='flatten'))

    # model.add(Dense(10, activation="softmax"))
    model.add(Dense(10, activation=None))
    conv_base.trainable = False

    model.load_weights('networks/imagenet/classifiers/imagenet_densenet121_weights.h5')

    return model,conv_base

def get_imagenet_vgg16():
    conv_base = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    model = Sequential()

    model.add(conv_base)
    model.add(Flatten(name='flatten'))

    model.add(Dense(10, activation="softmax"))
    # model.add(Dense(10, activation=None))
    conv_base.trainable = False

    model.load_weights('networks/imagenet/classifiers/imagenet_vgg16_weights.h5')

    return model,conv_base



def get_imagenet_inceptionv3():
    conv_base = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    model = Sequential()

    model.add(conv_base)
    model.add(Flatten(name='flatten'))

    # model.add(Dense(10, activation="softmax"))
    model.add(Dense(10, activation=None))
    conv_base.trainable = False

    model.load_weights('networks/imagenet/classifiers/imagenet_inceptionv3_weights.h5')

    return model,conv_base


def get_mnist_encoder():
    '''
    :return: an encoder without full connection layers
    '''
    inputs = Input(shape=(28, 28, 1))
    conv = Conv2D(filters=16, kernel_size=3, strides=1,activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv = Conv2D(filters=16, kernel_size=3, strides=1,activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    max_pool = MaxPooling2D((2, 2), (2, 2),padding="same")(conv)
    conv = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu',padding='same', kernel_initializer='he_normal')(max_pool)
    conv = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu',padding='same', kernel_initializer='he_normal')(conv)
    max_pool = MaxPooling2D((2, 2), (2, 2), padding="same")(conv)

    return Model(inputs=inputs, outputs=max_pool)


def get_mnist_full_connection_layers(vector_dimension):
    '''
    :return:  a model with the outputs of encoder as inputs and generate feature vectors
    '''
    inputs = Input(shape=(7,7,32))
    flat = Flatten()(inputs)
    feature_vector = Dense(vector_dimension, activation=None, kernel_initializer='he_normal')(flat)
    return Model(inputs=inputs, outputs=feature_vector)

def get_mnist_decoder(vector_dimension):
    '''
    :return: a decoder
    '''
    inputs = Input(shape=(vector_dimension,))
    fc = Dense(7 * 7 * 32, activation="relu", kernel_initializer='he_normal')(inputs)

    reshape_fc = Reshape((7, 7, 32))(fc)
    up_sampling = UpSampling2D(size=(2, 2))(reshape_fc)

    conv = Conv2D(filters=32, kernel_size=3, strides=1,activation='relu', padding='same', kernel_initializer='he_normal')(up_sampling)
    conv = Conv2D(filters=32, kernel_size=3, strides=1,activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    up_sampling = UpSampling2D(size=(2, 2))(conv)

    conv = Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', padding='same',
                  kernel_initializer='he_normal')(up_sampling)
    conv = Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', padding='same',
                  kernel_initializer='he_normal')(conv)
    outputs = Conv2D(filters=1, kernel_size=3, strides=1, padding='same',
                     activation='sigmoid', kernel_initializer='he_normal')(conv)
    return Model(inputs=inputs, outputs=outputs)

def get_discriminator_global(input_shape):
    inputs = Input(shape=input_shape)
    fc = Dense(256, activation="relu", kernel_initializer='he_normal')(inputs)
    fc = Dense(256, activation="relu", kernel_initializer='he_normal')(fc)
    fc = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(fc)

    return Model(inputs=inputs,outputs=fc)

def get_discriminator_local(input_shape):
    inputs = Input(shape=input_shape)
    conv = Conv2D(filters=256, kernel_size=1, strides=1, padding='same',
                  activation='relu', kernel_initializer='glorot_uniform')(inputs)
    conv = Conv2D(filters=256, kernel_size=1, strides=1, padding='same',
                  activation='relu', kernel_initializer='glorot_uniform')(conv)
    conv = Conv2D(filters=1, kernel_size=1, strides=1, padding='same',
                  activation='sigmoid', kernel_initializer='glorot_uniform')(conv)

    return Model(inputs=inputs,outputs=conv)

def get_discriminator_prior(input_shape):
    inputs = Input(shape=input_shape)
    fc = Dense(256, activation="relu", kernel_initializer='he_normal')(inputs)
    fc = Dense(256, activation="relu", kernel_initializer='he_normal')(fc)
    fc = Dense(1, activation=None, kernel_initializer='he_normal')(fc)

    return Model(inputs=inputs,outputs=fc)


def get_cifar10_encoder():
    inputs = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  activation='relu',
                  padding='same',
                  kernel_initializer='glorot_uniform')(inputs)

    conv = Conv2D(32, (3, 3),
                  strides=(1, 1),
                  padding='same',
                  activation='relu',
                  kernel_initializer='glorot_uniform')(conv)

    max_pool = MaxPooling2D((2, 2), (2, 2), padding="same")(conv)

    conv = Conv2D(64, (3, 3),
                  strides=(1,1),
                  padding='same',
                  activation='relu',
                  kernel_initializer='glorot_uniform')(max_pool)

    conv = Conv2D(64, (3, 3),
                  strides=(1, 1),
                  activation='relu',
                  padding='same',
                  kernel_initializer='glorot_uniform')(conv)

    max_pool = MaxPooling2D((2, 2), (2, 2), padding="same")(conv)


    return Model(inputs=inputs, outputs=max_pool)



def get_cifar10_full_connection_layers(vector_dimension):
    inputs = Input(shape=(8, 8, 64))

    flat = Flatten()(inputs)
    d = Dense(2048,activation='relu',kernel_initializer='he_normal')(flat)
    feature_vector = Dense(vector_dimension,activation=None,kernel_initializer='he_normal')(d)

    return Model(inputs=inputs, outputs=feature_vector)

def get_cifar10_decoder(vector_dimension):
    inputs = Input(shape=(vector_dimension,))

    d = Dense(2048,activation='relu',kernel_initializer='he_normal')(inputs)

    fc = Dense(8*8*64,activation="relu", kernel_initializer='he_normal')(d)

    reshape_fc = Reshape((8, 8, 64))(fc)
    up_sampling = UpSampling2D(size=(2, 2))(reshape_fc)

    conv = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same', kernel_initializer='glorot_uniform')(up_sampling)

    conv = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv)

    up_sampling = UpSampling2D(size=(2, 2))(conv)

    conv = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same', kernel_initializer='glorot_uniform')(
        up_sampling)

    conv = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv)


    outputs = Conv2D(filters=3, kernel_size=3, strides=1, padding='same',
                                     activation='sigmoid', kernel_initializer='glorot_uniform')(conv)
    return Model(inputs=inputs, outputs=outputs)



def get_imagenet_encoder():
    inputs = tf.keras.Input(shape=(224, 224, 3))

    conv = tf.keras.layers.Conv2D(filters=32,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  padding='same',
                                  activation="relu",
                                  kernel_initializer='glorot_uniform')(inputs)

    conv = tf.keras.layers.Conv2D(32 , (3, 3),
                                  padding='same',
                                  activation="relu",
                                  kernel_initializer='glorot_uniform')(conv)

    max_pooling = tf.keras.layers.MaxPool2D((2, 2), (2, 2),
                                            padding="same")(conv)

    conv = tf.keras.layers.Conv2D(64, (3, 3),
                                  padding='same',
                                  activation="relu",
                                  kernel_initializer='glorot_uniform')(max_pooling)
    conv = tf.keras.layers.Conv2D(64, (3, 3),
                                  padding='same',
                                  activation="relu",
                                  kernel_initializer='glorot_uniform')(conv)
    max_pooling = tf.keras.layers.MaxPool2D((2, 2), (2, 2),
                                            padding="same")(conv)
    conv = tf.keras.layers.Conv2D(128, (3, 3),
                                  padding='same',
                                  activation="relu",
                                  kernel_initializer='glorot_uniform')(max_pooling)
    conv = tf.keras.layers.Conv2D(128, (3, 3),
                                  padding='same',
                                  activation="relu",
                                  kernel_initializer='glorot_uniform')(conv)
    max_pooling = tf.keras.layers.MaxPool2D((2, 2), (2, 2),
                                            padding="same")(conv)
    conv = tf.keras.layers.Conv2D(128, (3, 3),
                                  padding='same',
                                  activation="relu",
                                  kernel_initializer='glorot_uniform')(max_pooling)
    conv = tf.keras.layers.Conv2D(128, (3, 3),
                                  padding='same',
                                  activation="relu",
                                  kernel_initializer='glorot_uniform')(conv)
    max_pooling = tf.keras.layers.MaxPool2D((2, 2), (2, 2),
                                            padding="same")(conv)

    return tf.keras.Model(inputs=inputs,outputs=max_pooling)

def get_imagenet_full_connection_layers(vector_dimension):
    inputs = tf.keras.Input(shape=(14,  14, 128))
    flat = tf.keras.layers.Flatten()(inputs)
    feature_vector = tf.keras.layers.Dense(vector_dimension,activation=None,kernel_initializer='he_normal')(flat)

    return tf.keras.Model(inputs=inputs, outputs=feature_vector)

def get_imagenet_decoder(vector_dimension):
    inputs = tf.keras.Input(shape=(vector_dimension,))
    fc = tf.keras.layers.Dense(14 * 14 * 128, activation="relu", kernel_initializer='he_normal')(inputs)
    reshape_fc = tf.keras.layers.Reshape((14, 14, 128))(fc)
    up_sampling = tf.keras.layers.UpSampling2D(size=(2, 2))(reshape_fc)

    conv = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer='glorot_uniform')(up_sampling)
    conv = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer='glorot_uniform')(conv)
    up_sampling = tf.keras.layers.UpSampling2D(size=(2, 2))(conv)
    conv = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer='glorot_uniform')(up_sampling)
    conv = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer='glorot_uniform')(conv)
    up_sampling = tf.keras.layers.UpSampling2D(size=(2, 2))(conv)
    conv = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer='glorot_uniform')(up_sampling)
    conv = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer='glorot_uniform')(conv)
    up_sampling = tf.keras.layers.UpSampling2D(size=(2, 2))(conv)
    conv = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer='glorot_uniform')(up_sampling)
    conv = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same',
                                  activation='relu', kernel_initializer='glorot_uniform')(conv)
    outputs = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1, padding='same',
                                     activation='sigmoid', kernel_initializer='glorot_uniform')(conv)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
