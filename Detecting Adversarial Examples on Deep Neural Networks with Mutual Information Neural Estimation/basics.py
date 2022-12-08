
import tensorflow as tf
import numpy as np


class AdamOptWrapper(tf.keras.optimizers.Adam):
    def __init__(self,
                 learning_rate=1e-4,
                 beta_1=0.5,
                 beta_2=0.999,
                 epsilon=1e-4,
                 amsgrad=False,
                 **kwargs):
        super(AdamOptWrapper, self).__init__(learning_rate, beta_1, beta_2, epsilon,
                                             amsgrad, **kwargs)


def global_loss(pos,neg):
    loss = - tf.reduce_mean(tf.math.log(pos + 1e-6) + tf.math.log(1 - neg + 1e-6))
    return loss

def local_loss(pos,neg):
    loss = -tf.reduce_mean(tf.math.log(tf.reduce_mean(pos,[1,2,3]) + 1e-6) + tf.math.log(1-tf.reduce_mean(neg,[1,2,3])+ 1e-6))
    return loss


def discriminator_loss(fake, real):
    f_loss = tf.reduce_mean(fake)
    r_loss = tf.reduce_mean(real)
    return f_loss - r_loss

def generator_loss(fake):
    f_loss = -tf.reduce_mean(fake)
    return f_loss

def MSE(real_image, fake_image):
    loss = tf.reduce_mean(tf.square(real_image - fake_image))
    return loss



def gradient_penalty(batch_size,f, real, fake):
    alpha = tf.random.uniform([batch_size, 1], 0., 1.)
    diff = fake - real
    inter = real + (alpha * diff)

    with tf.GradientTape() as t:
        t.watch(inter)
        pred = f(inter)
    grad = t.gradient(pred, [inter])[0]

    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]))
    gp = tf.reduce_mean((slopes - 1.)**2)
    return gp


def get_testing_data(x_test,x_adv_test,y_test,classifier):
    preds = classifier.predict(x_test)
    y1 = np.reshape(y_test,(y_test.shape[0],))
    inds_correct = np.where(preds.argmax(axis=1) == y1)[0]
    x_adv_test = x_adv_test[inds_correct]
    y_test = y_test[inds_correct]
    y1= np.reshape(y_test,(y_test.shape[0],))
    x_test = x_test[inds_correct]
    preds_adv = classifier.predict(x_adv_test)
    inds_correct = np.where(preds_adv.argmax(axis=1) == y1)[0]
    x_adv = np.delete(x_adv_test,inds_correct,axis=0)
    x = np.delete(x_test,inds_correct,axis=0)
    y = np.delete(y_test,inds_correct,axis=0)

    return x,x_adv,y

